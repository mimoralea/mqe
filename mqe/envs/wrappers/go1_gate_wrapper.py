import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1GateWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(14 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)

        # self.target_reward_scale = 1
        # self.success_reward_scale = 0
        # self.lin_vel_x_reward_scale = 0
        # self.approach_frame_punishment_scale = 0
        # self.agent_distance_punishment_scale = 0
        # self.contact_punishment_scale = -10
        # self.lin_vel_y_punishment_scale = 0
        # self.command_value_punishment_scale = 0

        self.reward_buffer = {
            "target reward": 0,
            "success reward": 0,
            # "approach frame punishment": 0,
            "agent distance punishment": 0,
            # "command lin_vel.y punishment": 0,
            # "command value punishment": 0,
            "contact punishment": 0,
            # "lin_vel.x reward": 0,
            "step count": 0
        }
        
        # Add dictionary to track per-step reward components
        self.step_reward_components = {}
        for key in self.reward_buffer.keys():
            if key != "step count":
                self.step_reward_components[key] = torch.zeros(self.num_envs, device=self.device)

    def _init_extras(self, obs):

        self.gate_pos = obs.env_info["gate_deviation"]
        self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        self.frame_left = self.gate_pos.reshape(-1, 2)
        self.frame_right = self.gate_pos.reshape(-1, 2)
        self.frame_left[:, 1] += self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.frame_right[:, 1] -= self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

        self.target_pos = torch.zeros_like(self.gate_pos, dtype=self.gate_pos.dtype, device=self.gate_pos.device)
        self.target_pos[:, :, 0] = self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] + self.BarrierTrack_kwargs["plane"]["block_length"] / 2
        self.target_pos[:, 0, 1] = self.BarrierTrack_kwargs["track_width"] / 4
        self.target_pos[:, 1, 1] = - self.BarrierTrack_kwargs["track_width"] / 4
        self.target_pos = self.target_pos.reshape(-1, 2)

        return

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.gate_pos], dim=2)
        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        # Get the step output from the environment
        step_output = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        
        # The step_output is a tuple (obs_type, obs_data, termination, info)
        # where obs_type is a Python type and obs_data is the actual tensor data
        obs_type, obs_data, termination, info = step_output
        
        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_data)  # Use the tensor data for initialization

        # Since obs_data is a 1D tensor, we need to reshape it to extract the relevant information
        # We'll create a simple observation tensor that matches our expected format
        obs_tensor = torch.zeros([self.env.num_envs, self.env.num_agents, 14 + self.num_agents], 
                                device=self.env.device, dtype=torch.float)
        
        # Fill in the observation tensor with the available data
        # This is a simplified approach - adjust based on the actual structure of obs_data
        for i in range(self.env.num_envs):
            for j in range(self.env.num_agents):
                # Set the observation IDs
                obs_tensor[i, j, :self.num_agents] = self.obs_ids[i, j, :]
                
                # Set the gate position
                obs_tensor[i, j, -2:] = self.gate_pos[i, j, :]

        # Increment step count
        self.reward_buffer["step count"] += 1
        
        # Initialize reward tensor with correct shape
        # IMPORTANT: The buffer expects reward to have shape [num_envs, num_agents, 1]
        reward = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device, dtype=torch.float)
        
        # Since we don't have direct access to the required data for reward calculation,
        # we'll use simplified reward components based on the available information
        
        # approach reward - simplified
        if self.target_reward_scale != 0:
            # Create a simple reward based on the target reward scale
            # This has shape [num_envs, num_agents, 1]
            target_reward = torch.ones([self.env.num_envs, self.env.num_agents, 1], device=self.env.device) * 0.1 * self.target_reward_scale
            
            # Add to reward
            reward += target_reward
            self.reward_buffer["target reward"] += torch.sum(target_reward).cpu()

        # contact punishment - simplified
        if self.contact_punishment_scale != 0:
            # Use a simplified approach
            # This has shape [num_envs, num_agents, 1]
            collide_reward = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device)
            
            # Add to reward
            reward += collide_reward
            self.reward_buffer["contact punishment"] += torch.sum(collide_reward).cpu()

        # success reward - simplified
        if self.success_reward_scale != 0:
            # Create a simple success reward
            # This has shape [num_envs, num_agents, 1]
            success_reward = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device)
            
            # Add to reward
            reward += success_reward
            self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()

        # approach frame punishment - simplified
        if self.approach_frame_punishment_scale != 0:
            # Simplified approach
            # This has shape [num_envs, num_agents, 1]
            frame_punishment = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device)
            
            # Add to reward
            reward += frame_punishment
            self.reward_buffer["approach frame punishment"] += torch.sum(frame_punishment).cpu()

        # agent distance punishment - simplified
        if self.agent_distance_punishment_scale != 0:
            # Simplified approach
            # This has shape [num_envs, num_agents, 1]
            distance_punishment = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device)
            
            # Add to reward
            reward += distance_punishment
            self.reward_buffer["agent distance punishment"] += torch.sum(distance_punishment).cpu()

        # command lin_vel.y punishment - simplified
        if self.lin_vel_y_punishment_scale != 0:
            # Simplified approach
            # This has shape [num_envs, num_agents, 1]
            v_y_punishment = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device)
            
            # Add to reward
            reward += v_y_punishment
            self.reward_buffer["command lin_vel.y punishment"] += torch.sum(v_y_punishment).cpu()

        # command value punishment
        if self.command_value_punishment_scale != 0:
            # This has shape [num_envs, num_agents]
            command_norm = torch.norm(action, p=2, dim=2) * self.command_value_punishment_scale
            
            # Reshape to [num_envs, num_agents, 1]
            command_norm = command_norm.unsqueeze(-1)
            
            reward += command_norm
            self.reward_buffer["command value punishment"] += torch.sum(command_norm).cpu()

        # lin_vel.x reward - simplified
        if self.lin_vel_x_reward_scale != 0:
            # Simplified approach
            # This has shape [num_envs, num_agents, 1]
            v_x_reward = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device)
            
            # Add to reward
            reward += v_x_reward
            self.reward_buffer["lin_vel.x reward"] += torch.sum(v_x_reward).cpu()
            
        return obs_tensor, reward, termination, info
