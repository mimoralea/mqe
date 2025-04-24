import gym
from gym import spaces
import numpy as np
import torch
from enum import IntEnum, Enum, auto
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

"""
Go1Gate Environment
==================

Description
----------
The Go1Gate environment is a multi-agent reinforcement learning environment where Unitree Go1 robots
need to navigate through a gate. The environment is designed to train agents to coordinate and navigate
efficiently through a constrained space while avoiding collisions with walls and other agents.

Observation Space
----------------
Each agent receives a 13-dimensional observation vector:

- Indices 0-2: Robot orientation (roll, pitch, yaw) in radians
- Indices 3-4: Relative position (x, y) to the other agent
- Index 5: Binary flag indicating if teammate has crossed the gate (1.0 = crossed)
- Indices 6-7: Relative position (x, y) to the gate
- Index 8: Binary flag indicating if this agent has crossed the gate (1.0 = crossed)
- Indices 9-12: Distances to the walls (left, right, back, front)
  - Left wall at y = -1.5
  - Right wall at y = 1.5
  - Back wall at x = 0
  - Front wall at x = 6.1

Action Space
-----------
The action space is a 3-dimensional continuous space with values in the range [-1, 1]:
- Forward/backward movement
- Left/right movement
- Yaw rotation

These actions are scaled by [2, 0.5, 0.5] respectively before being applied to the robot.

Termination Conditions
--------------------
An episode terminates when any of the following conditions are met:
1. All agents successfully cross the gate (gate)
2. An agent collides with another agent or obstacle (collision)
3. An agent exceeds roll or pitch thresholds (roll/pitch)
4. The episode reaches the maximum time limit (timeout)

Reward Structure
--------------
Terminal Rewards:
- Gate crossing (GATE = 10.0): Awarded when all agents successfully cross the gate
- Timeout (TIMEOUT = -10.0): Penalty for not completing the task within the time limit
- Collision (COLLISION = -10.0): Penalty for colliding with obstacles or other agents
- Roll/Pitch (ROLL/PITCH = -10.0): Penalty for excessive roll or pitch angles

Shaping Rewards:
1. X-axis Shaping: Encourages progress toward the gate along the x-axis
   - Progression (0.01): Given when moving toward the gate or when already past the gate
   - Regression (-0.02): Penalty for moving away from the gate

2. Y-axis Shaping: Encourages movement toward the center of the gate along the y-axis
   - Progression (0.01): Given when moving toward the center, when very close to center,
     or when already past the gate
   - Regression (-0.02): Penalty for moving away from the center

3. Wall Avoidance Shaping: Discourages proximity to walls
   - Wall penalty (-0.01): Applied when an agent is within 0.15 units of any wall
   - No penalty (0.0): When the agent maintains a safe distance from all walls
"""

class ObservationIndex(IntEnum):
    """Indices for the observation vector components in the Go1Gate environment.

    These indices define the structure of the observation space, which includes
    information about the robot's orientation, relative positions to other agents
    and the gate, gate crossing status, and distances to walls.
    """
    ROLL = 0                # Roll angle of the robot in radians
    PITCH = 1               # Pitch angle of the robot in radians
    YAW = 2                 # Yaw angle of the robot in radians
    OTHER_REL_X = 3         # Relative X position to the other agent
    OTHER_REL_Y = 4         # Relative Y position to the other agent
    OTHER_CROSSED_GATE = 5  # Binary flag indicating if teammate has crossed the gate (1.0 = crossed)
    GATE_REL_X = 6          # Relative X position to the gate
    GATE_REL_Y = 7          # Relative Y position to the gate
    MY_CROSSED_GATE = 8     # Binary flag indicating if this agent has crossed the gate (1.0 = crossed)
    LEFT_WALL_DIST = 9      # Distance to the left wall (y = -1.5)
    RIGHT_WALL_DIST = 10    # Distance to the right wall (y = 1.5)
    BACK_WALL_DIST = 11     # Distance to the back wall (x = 0)
    FRONT_WALL_DIST = 12    # Distance to the front wall (x = 6.1)


class RewardScale:
    """Reward scale values for different components of the reward function.

    These values define the magnitude of rewards and penalties for various
    events and behaviors in the Go1Gate environment.
    """
    GATE = 10.0                # Reward for successfully passing through the gate
    TIMEOUT = -10.0            # Penalty for timing out (not completing the task in time)
    COLLISION = -10.0          # Penalty for colliding with obstacles or other agents
    ROLL = -10.0               # Penalty for excessive roll angle
    PITCH = -10.0              # Penalty for excessive pitch angle
    PROGRESSION_SHAPING = 0.01 # Small reward for moving toward the goal (shaping)
    REGRESSION_SHAPING = -0.02 # Small penalty for moving away from the goal (shaping)
    WALL_SHAPING = -0.01       # Small penalty for being too close to walls (shaping)
    UNDEFINED = 0.0            # Default reward for undefined conditions


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
            "wall_shaping": None,
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

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,)
        )
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
        # Add these lines at the end of _init_extras
        self.left_wall_y = -self.BarrierTrack_kwargs["track_width"] / 2
        self.right_wall_y = self.BarrierTrack_kwargs["track_width"] / 2
        self.back_wall_x = 0.0  # Origin
        self.front_wall_x = (self.BarrierTrack_kwargs["init"]["block_length"] +
                            self.BarrierTrack_kwargs["gate"]["block_length"] +
                            self.BarrierTrack_kwargs["plane"]["block_length"] +
                            self.BarrierTrack_kwargs["wall"]["block_length"])

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
        next_obs[:, :, ObservationIndex.ROLL:ObservationIndex.YAW+1] = base_rpy_reshaped

        for agent_idx in range(self.env.num_agents):
            # get absolute positions
            other_agent_idx = (agent_idx + 1) % self.env.num_agents
            my_pos = base_pos_reshaped[:, agent_idx, :2]
            other_pos = base_pos_reshaped[:, other_agent_idx, :2]
            gate_pos = self.gate_pos[:, agent_idx, :2]

            # Calculate relative position (x,y) to the other agent
            other_rel_pos = other_pos - my_pos
            next_obs[:, agent_idx, ObservationIndex.OTHER_REL_X:ObservationIndex.OTHER_REL_Y+1] = other_rel_pos

            # Has my teammate crossed the gate?
            other_gate_crossed = (base_pos_reshaped[:, other_agent_idx, 0] > (self.gate_distance+0.5)).float()
            next_obs[:, agent_idx, ObservationIndex.OTHER_CROSSED_GATE] = other_gate_crossed

            # Calculate relative position (x,y) to the gate
            gate_rel_pos = gate_pos - my_pos
            next_obs[:, agent_idx, ObservationIndex.GATE_REL_X:ObservationIndex.GATE_REL_Y+1] = gate_rel_pos

            # Have I crossed the gate?
            my_gate_crossed = (base_pos_reshaped[:, agent_idx, 0] > (self.gate_distance+0.5)).float()
            next_obs[:, agent_idx, ObservationIndex.MY_CROSSED_GATE] = my_gate_crossed

            # Calculate distances to walls
            # Distance to left wall
            left_wall_distance = my_pos[:, 1] - self.left_wall_y
            next_obs[:, agent_idx, ObservationIndex.LEFT_WALL_DIST] = left_wall_distance

            # Distance to right wall
            right_wall_distance = self.right_wall_y - my_pos[:, 1]
            next_obs[:, agent_idx, ObservationIndex.RIGHT_WALL_DIST] = right_wall_distance

            # Distance to back wall
            back_wall_distance = my_pos[:, 0] - self.back_wall_x
            next_obs[:, agent_idx, ObservationIndex.BACK_WALL_DIST] = back_wall_distance

            # Distance to front wall
            front_wall_distance = self.front_wall_x - my_pos[:, 0]
            next_obs[:, agent_idx, ObservationIndex.FRONT_WALL_DIST] = front_wall_distance

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
        g = next_obs[:, :, ObservationIndex.MY_CROSSED_GATE].to(torch.bool)
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
        self.rewards["gate"][batch_indices, :, self.ts] = self.dones["gate"][batch_indices, :, self.ts] * RewardScale.GATE
        self.rewards["timeout"][batch_indices, :, self.ts] = self.dones["timeout"][batch_indices, :, self.ts] * RewardScale.TIMEOUT
        self.rewards["collision"][batch_indices, :, self.ts] = self.dones["collision"][batch_indices, :, self.ts] * RewardScale.COLLISION
        self.rewards["pitch"][batch_indices, :, self.ts] = self.dones["pitch"][batch_indices, :, self.ts] * RewardScale.PITCH
        self.rewards["roll"][batch_indices, :, self.ts] = self.dones["roll"][batch_indices, :, self.ts] * RewardScale.ROLL
        self.rewards["undefined"][batch_indices, :, self.ts] = self.dones["undefined"][batch_indices, :, self.ts] * RewardScale.UNDEFINED

        # Calculate change in x- and y-position for each agent
        delta_xy = next_obs[:, :, ObservationIndex.GATE_REL_X:ObservationIndex.GATE_REL_Y+1] - self.obs[:, :, ObservationIndex.GATE_REL_X:ObservationIndex.GATE_REL_Y+1]

        # Get x component while preserving batch dimensions
        delta_x = delta_xy[:, :, 0]

        # X-axis shaping: reward for moving towards the gate
        # If delta_x < 0, the agent is moving towards the gate (gate_rel_pos[0] is decreasing)
        # Give progression reward if:
        # 1. Moving towards the gate (delta_x < 0), or
        # 2. Already cleared the gate (g)
        # Otherwise, give regression penalty
        x_axis_shaping = torch.where(
            (delta_x < 0) | g,  # If moving towards gate OR already cleared gate
            RewardScale.PROGRESSION_SHAPING,  # Reward for progress or maintaining goal
            RewardScale.REGRESSION_SHAPING  # Penalty for moving away from goal
        )
        self.rewards["x_axis_shaping"][batch_indices, :, self.ts] = x_axis_shaping

        # Calculate y-axis shaping based on movement towards center of gate
        # next_y is the y-component of the relative position to the gate
        next_y = next_obs[:, :, ObservationIndex.GATE_REL_Y]
        delta_y = next_y.abs() - self.obs[:, :, ObservationIndex.GATE_REL_Y].abs()  # Change in absolute distance to center

        # Y-axis shaping: reward for moving towards the center of the gate
        # Give progression reward if:
        # 1. Moving towards the center (delta_y < 0), or
        # 2. Very close to the center (next_y.abs() < 0.15), or
        # 3. Already cleared the gate (g)
        # Otherwise, give regression penalty
        y_axis_shaping = torch.where(
            (delta_y < 0) | (next_y.abs() < 0.15) | g,  # Progress OR at goal OR cleared gate
            RewardScale.PROGRESSION_SHAPING,  # Reward for progress or maintaining goal
            RewardScale.REGRESSION_SHAPING  # Penalty for moving away from goal
        )
        self.rewards["y_axis_shaping"][batch_indices, :, self.ts] = y_axis_shaping

        # Calculate wall avoidance shaping reward
        # Get distances to all walls from the observation vector
        left_wall_distance = next_obs[:, :, ObservationIndex.LEFT_WALL_DIST]
        right_wall_distance = next_obs[:, :, ObservationIndex.RIGHT_WALL_DIST]
        back_wall_distance = next_obs[:, :, ObservationIndex.BACK_WALL_DIST]
        front_wall_distance = next_obs[:, :, ObservationIndex.FRONT_WALL_DIST]

        # Check if agent is too close to any wall (less than 0.15 units)
        too_close_to_left = left_wall_distance < 0.15
        too_close_to_right = right_wall_distance < 0.15
        too_close_to_back = back_wall_distance < 0.15
        too_close_to_front = front_wall_distance < 0.15

        # Combine all conditions - if any is true, agent is too close to a wall
        too_close_to_any_wall = too_close_to_left | too_close_to_right | too_close_to_back | too_close_to_front

        # Apply wall shaping reward: -0.01 if too close, 0 otherwise
        wall_shaping = torch.where(
            too_close_to_any_wall,
            RewardScale.WALL_SHAPING,  # Negative reward for being too close
            0.0  # No penalty if not too close
        )
        self.rewards["wall_shaping"][batch_indices, :, self.ts] = wall_shaping

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
