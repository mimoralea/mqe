import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from isaacgym.torch_utils import get_euler_xyz

class Go1GateWrapper(EmptyWrapper):
    def __init__(self, env):
        """
        Initialize the Go1GateWrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)

        # Store the original environment
        self.env = env
        self.device = self.env.device

        # Get max episode length for buffer sizing
        self.max_episode_length = int(self.max_episode_length) + 1

        # Initialize reward buffers for tracking rewards
        self.rewards = {
            "roll": None,
            "pitch": None,
            "collision": None,
            "timeout": None,
            "gate": None,
            "x_axis_shaping": None,
            "y_axis_shaping": None,
            "crossed_shaping": None,
            "undefined": None,
        }

        # Initialize terminal buffers for tracking termination reasons
        self.dones = {
            "roll": None,
            "pitch": None,
            "collision": None,
            "timeout": None,
            "gate": None,
            "undefined": None,
        }

        # Initialize episode tracking variables
        self.ts = None

        # Initialize extras
        self._init_extras(None)

        # Reward scales
        self.gate_reward = 10.0
        self.timeout_reward = 0.0
        self.collision_reward = 0.0
        self.roll_reward = 0.0
        self.pitch_reward = 0.0
        self.cleared_shaping = 0.1
        self.closer_shaping = 0.01
        self.further_shaping = 0.0
        self.crossed_shaping = 0.0
        self.undefined_reward = 0.0

        # Update observation space to the enhanced 8-dimensional space
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

    def _init_extras(self, obs):
        # We no longer need the observation IDs since we're not using them in the simplified space
        # But we'll keep track of the gate position for reward calculation and observation construction

        # Initialize the gate position
        self.gate_pos = torch.zeros((self.env.num_envs, self.num_agents, 3), device=self.device)

        # Handle the case where obs is a tensor or an object with attributes
        if isinstance(obs, torch.Tensor):
            # If obs is a tensor, we need to extract gate_deviation from info
            # For now, we'll initialize with default values
            gate_pos_2d = torch.zeros((self.env.num_envs, 2), device=self.device)
            gate_pos_2d[:, 0] = self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        elif hasattr(obs, 'env_info') and "gate_deviation" in obs.env_info:
            gate_pos_2d = obs.env_info["gate_deviation"]
            gate_pos_2d[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        else:
            # Default initialization if gate_deviation is not available
            gate_pos_2d = torch.zeros((self.env.num_envs, 2), device=self.device)
            gate_pos_2d[:, 0] = self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2

        # Expand gate position to 3D and repeat for each agent
        self.gate_pos[:, :, :2] = gate_pos_2d.unsqueeze(1).repeat(1, self.num_agents, 1)

        # Calculate frame positions (for reward calculation)
        self.frame_left = gate_pos_2d.clone()
        self.frame_right = gate_pos_2d.clone()
        self.frame_left[:, 1] += self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.frame_right[:, 1] -= self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.gate_distance = gate_pos_2d[:, 0]

        # Initialize target position with proper dimensions [num_envs, num_agents, 2]
        self.target_pos = torch.zeros((self.env.num_envs, self.num_agents, 2), device=self.device)
        # Set x coordinate (same for all agents)
        self.target_pos[:, :, 0] = self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] + self.BarrierTrack_kwargs["plane"]["block_length"] / 2
        # Set y coordinate (slightly different for each agent to encourage different behaviors)
        self.target_pos[:, 0, 1] = self.BarrierTrack_kwargs["track_width"] / 4
        self.target_pos[:, 1, 1] = -self.BarrierTrack_kwargs["track_width"] / 4

    def _process_observations(self, obs):
        """
        Process observations from the environment.

        Args:
            obs: Raw observations from the environment

        Returns:
            Processed observations
        """

        # Construct the observation space
        # 0. Create empty tensor for the new observations
        next_obs = torch.zeros((self.env.num_envs, self.env.num_agents, self.observation_space.shape[0]), device=self.device)

        # Extract base position and orientation
        if isinstance(obs, torch.Tensor):
            if obs.dim() == 1:
                # Create default observations
                base_pos = torch.zeros((self.env.num_envs * self.env.num_agents, 3), device=self.device)
                base_rpy = torch.zeros((self.env.num_envs * self.env.num_agents, 3), device=self.device)
            else:
                # Extract from tensor
                base_pos = obs[:, :3].reshape(self.env.num_envs * self.env.num_agents, 3)
                base_rpy = obs[:, 3:6].reshape(self.env.num_envs * self.env.num_agents, 3)
        else:
            # Extract from object with attributes
            base_pos = obs.base_pos
            base_rpy = obs.base_rpy

        # Reshape for easier access
        base_pos_reshaped = base_pos.reshape(self.env.num_envs, self.env.num_agents, 3)
        base_rpy_reshaped = base_rpy.reshape(self.env.num_envs, self.env.num_agents, 3)

        # Base robot orientation (3D: roll, pitch, yaw)
        next_obs[:, :, 0:3] = base_rpy_reshaped

        for agent_idx in range(self.env.num_agents):
            # get absolute positions
            other_agent_idx = (agent_idx + 1) % self.env.num_agents
            my_pos = base_pos_reshaped[:, agent_idx, :2]
            other_pos = base_pos_reshaped[:, other_agent_idx, :2]
            gate_pos = self.gate_pos[:, agent_idx, :2]

            # Calculate relative position (x,y) to the other agent
            other_rel_pos = other_pos - my_pos
            next_obs[:, agent_idx, 3:5] = other_rel_pos

            # Has my teammate crossed the gate?
            other_gate_crossed = (base_pos_reshaped[:, other_agent_idx, 0] > (self.gate_distance+0.3)).float()
            next_obs[:, agent_idx, 5] = other_gate_crossed

            # Calculate relative position (x,y) to the gate
            gate_rel_pos = gate_pos - my_pos
            next_obs[:, agent_idx, 6:8] = gate_rel_pos

            # Have I crossed the gate?
            my_gate_crossed = (base_pos_reshaped[:, agent_idx, 0] > (self.gate_distance+0.3)).float()
            next_obs[:, agent_idx, 9] = my_gate_crossed

        return next_obs

    def reset(self, **kwargs):
        """Reset the environment and clear buffers."""
        # Initialize reward buffers for tracking rewards on a per-robot basis
        # Structure: [num_envs, num_agents, max_episode_length]
        for reward in self.rewards:
            self.rewards[reward] = torch.zeros((self.env.num_envs, self.env.num_agents, self.max_episode_length), device=self.device, dtype=torch.float32)

        # Initialize terminal buffers for tracking termination reasons on a per-robot basis
        # Structure: [num_envs, num_agents, max_episode_length]
        for done in self.dones:
            self.dones[done] = torch.zeros((self.env.num_envs, self.env.num_agents, self.max_episode_length), device=self.device, dtype=torch.bool)

        # Initialize episode tracking variables
        self.ts = torch.zeros((self.env.num_envs,), device=self.device, dtype=torch.int64)

        # Reset the parent environment
        parent_obs = self.env.reset() # this reset_idx on all envs in parent class

        # now reset the buffers of this class: rewards, dones, and ts
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Initialize extras
        self._init_extras(None)

        # Process observations
        self.obs = self._process_observations(parent_obs)
        return self.obs

    def step(self, action):

        # Reset the environment first
        if len(self.reset_ids) > 0:
            # Reset these environments using our own reset_idx method
            # This will properly reset both the parent environment and our wrapper's state
            self.reset_idx(self.reset_ids)

        # PROCESS ACTIONS
        # PROCESS ACTIONS
        # PROCESS ACTIONS
        # action = torch.from_numpy(action).to(self.device).clip(-1, 1)
        action = torch.clip(action, -1, 1)
        upstream_step_output = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        parent_obs, parent_rewards, parent_dones, parent_infos = upstream_step_output

        # PROCESS OBSERVATIONS
        # PROCESS OBSERVATIONS
        # PROCESS OBSERVATIONS
        next_obs = self._process_observations(parent_obs)

        # CALCULATE TERMINATION CONDITIONS
        # CALCULATE TERMINATION CONDITIONS
        # CALCULATE TERMINATION CONDITIONS
        # Store termination reasons in the terminal buffer
        batch_indices = torch.arange(self.env.num_envs, device=self.device)
        g = next_obs[:, :, 9].to(torch.bool)
        ga = g.all(1).unsqueeze(1).repeat(1, self.env.num_agents)
        t = self.env.time_out_buf.unsqueeze(1).repeat(1, self.env.num_agents)
        c = self.env.collide_buf_each
        p = self.env.p_term_buff_each
        r = self.env.r_term_buff_each
        assert torch.all(self.ts < self.max_episode_length), f"ts={self.ts}, max_episode_length={self.max_episode_length}"
        self.dones["gate"][batch_indices, :, self.ts] = ga
        self.dones["timeout"][batch_indices, :, self.ts] = t & ~ga
        self.dones["collision"][batch_indices, :, self.ts] = c & ~ga & ~t
        self.dones["pitch"][batch_indices, :, self.ts] = p & ~ga & ~t & ~c
        self.dones["roll"][batch_indices, :, self.ts] = r & ~ga & ~t & ~c & ~p

        # Check if any done condition is true for each environment
        any_done = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        for done in self.dones:
            # First check if any agent is done for this condition across all environments
            # Use batch_indices to access the correct timestep for each environment
            any_agent = self.dones[done][batch_indices, :, self.ts].any(1)
            any_done = any_done | any_agent

        # Assert that if parent_done is true, we've captured it in our self.dones
        # This ensures we're not missing any termination conditions
        # If parent_dones is true, then any_done should also be true
        # parent_dones => any_done is equivalent to ~parent_dones | any_done
        # assert torch.all(~parent_dones | any_done), "Some environments have parent_done=True but no done condition is set in self.dones"
        u = (~any_done & parent_dones).unsqueeze(1).repeat(1, self.env.num_agents)
        self.dones["undefined"][batch_indices, :, self.ts] = u
        any_done = any_done | u.any(1)

        # Now handle the delayed reset for environments that cleared the gate
        # This happens after all rewards and termination conditions have been processed
        self.reset_ids = torch.where(any_done)[0]

        # CALCULATE REWARDS
        # CALCULATE REWARDS
        # CALCULATE REWARDS
        # Now calculate rewards based on the termination conditions
        self.rewards["gate"][batch_indices, :, self.ts] = self.dones["gate"][batch_indices, :, self.ts] * self.gate_reward
        self.rewards["timeout"][batch_indices, :, self.ts] = self.dones["timeout"][batch_indices, :, self.ts] * self.timeout_reward
        self.rewards["collision"][batch_indices, :, self.ts] = self.dones["collision"][batch_indices, :, self.ts] * self.collision_reward
        self.rewards["pitch"][batch_indices, :, self.ts] = self.dones["pitch"][batch_indices, :, self.ts] * self.pitch_reward
        self.rewards["roll"][batch_indices, :, self.ts] = self.dones["roll"][batch_indices, :, self.ts] * self.roll_reward
        self.rewards["undefined"][batch_indices, :, self.ts] = self.dones["undefined"][batch_indices, :, self.ts] * self.undefined_reward

        # Calculate change in x-position for each agent
        delta_xy = next_obs[:, :, 6:8] - self.obs[:, :, 6:8]

        # Get x component while preserving batch dimensions
        delta_x = delta_xy[:, :, 0]
        x_axis_shaping = torch.where(
            delta_x < 0,
            self.closer_shaping,
            self.further_shaping
        )
        x_axis_shaping = torch.where(
            g,
            self.cleared_shaping,
            x_axis_shaping
        )
        self.rewards["x_axis_shaping"][batch_indices, :, self.ts] = x_axis_shaping

        # # Calculate change in y-position for each agent
        # next_y = next_obs[:, :, 7]
        # delta_y = next_y - self.obs[:, :, 7]
        # y_axis_shaping = torch.where(
        #     next_y.abs() < 0.1,
        #     self.closer_shaping,
        #     self.further_shaping
        # )
        # y_axis_shaping = torch.where(
        #     delta_y > 0,
        #     self.closer_shaping,
        #     self.further_shaping
        # )
        # y_axis_shaping = torch.where(
        #     g,
        #     self.cleared_shaping,
        #     y_axis_shaping
        # )
        # self.rewards["y_axis_shaping"][batch_indices, :, self.ts] = y_axis_shaping
        self.rewards["y_axis_shaping"][batch_indices, :, self.ts] = 0.0

        # Add shaping for crossing the gate, while you wait for the teammate
        self.rewards["crossed_shaping"][batch_indices, :, self.ts] = g * self.crossed_shaping

        # Create per agent sum of rewards, shape [num_envs, num_agents]
        agent_rewards = torch.zeros((self.env.num_envs, self.env.num_agents), device=self.device)

        # Sum the reward components for the current timestep using batch_indices
        for key in self.rewards:
            agent_rewards += self.rewards[key][batch_indices, :, self.ts]

        self.obs = next_obs        # Increment episode step counts
        self.ts += 1

        return self.obs, agent_rewards, any_done, parent_infos

    def reset_idx(self, env_indices):
        """Reset specific environment indices and their corresponding buffers."""
        # Call the environment's reset_idx method
        self.env.reset_idx(env_indices)

        # Reset reward buffers for these environments
        for key in self.rewards:
            self.rewards[key][env_indices] = torch.zeros((self.env.num_agents, self.max_episode_length), device=self.device, dtype=torch.float32)

        # Reset terminal buffers for these environments
        for key in self.dones:
            self.dones[key][env_indices] = torch.zeros((self.env.num_agents, self.max_episode_length), device=self.device, dtype=torch.bool)

        # Reset episode tracking variables for these environments
        self.ts[env_indices] = 0
