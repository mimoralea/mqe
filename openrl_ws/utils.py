from __future__ import print_function, division, absolute_import

from typing import Any, Dict, Optional, Union
import isaacgym

import numpy as np
import torch
import gym
from gym import spaces

from mqe.envs.utils import make_mqe_env

from openrl.configs.config import create_config_parser
from isaacgym import gymutil
from typing import List
from openrl.configs.utils import ProcessYamlAction

from abc import ABC, abstractmethod
import math
import numpy as np
import argparse
from bisect import bisect

from isaacgym import gymapi
from isaacgym.gymutil import parse_device_str

from mqe.envs.go1.go1_config import Go1Cfg
from openrl.envs.vec_env import BaseVecEnv

def make_env(args, custom_cfg=None, single_agent=False):

    env, env_cfg = make_mqe_env(args.task, args, custom_cfg=custom_cfg)

    if single_agent:
        env = SingleAgentWrapper(env)

    return mqe_openrl_wrapper(env), env_cfg

class mqe_openrl_wrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.agent_num = self.env.num_agents
        self.parallel_env_num = self.env.num_envs
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Initialize episode reward tracking - use the same device as the environment
        self.device = self.env.device
        self.episode_rewards = torch.zeros(self.parallel_env_num, device=self.device)
        self.episode_lengths = torch.zeros(self.parallel_env_num, device=self.device)
        self.episode_count = 0
        self.total_steps = 0
        
        # Logger reference (will be set by the driver)
        self.logger = None
        # Global step counter (will be updated by the driver)
        self.global_step = 0

    def reset(self, **kwargs):
        """Reset all environments."""
        obs = self.env.reset()
        return obs.cpu().numpy()

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        actions = torch.from_numpy(0.5 * actions).to(self.device).clip(-1, 1)

        obs, reward, termination, info = self.env.step(actions)
        
        # Update global step if provided in extra_data
        if extra_data is not None and "global_step" in extra_data:
            self.global_step = extra_data["global_step"]
        
        # Handle reward shape correctly - if reward is [num_envs, num_agents, 1], reshape it
        if reward.dim() == 3 and reward.shape[2] == 1:
            # Reshape from [num_envs, num_agents, 1] to [num_envs, num_agents]
            reward = reward.squeeze(-1)
        
        # Track episode rewards - make sure to keep everything on the same device
        self.episode_rewards += reward.sum(dim=1)  # Keep on GPU
        self.episode_lengths += 1
        self.total_steps += 1
        
        # Convert tensors to numpy for the environment interface
        obs = obs.cpu().numpy()
        
        # Make sure rewards have the correct shape for the buffer: [num_envs, num_agents, 1]
        # If reward is [num_envs, num_agents], add the extra dimension
        if reward.dim() == 2:
            rewards = reward.unsqueeze(-1).cpu().numpy()
        else:
            rewards = reward.cpu().unsqueeze(-1).numpy()
            
        dones = termination.cpu().unsqueeze(-1).repeat(1, self.agent_num).numpy().astype(bool)
        
        # Create info dictionaries and track completed episodes
        infos = []
        for i in range(dones.shape[0]):
            info_dict = {}
            # If the episode is done, add the episode reward to the info
            if termination[i]:
                # Increment episode counter
                self.episode_count += 1
                
                # Get episode stats - move to CPU only when needed for logging
                episode_reward = self.episode_rewards[i].item()  # .item() handles the device conversion
                episode_length = self.episode_lengths[i].item()
                
                # Add to info dict for the driver
                info_dict["episode"] = {
                    "r": episode_reward,
                    "l": episode_length,
                    "t": self.global_step,  # Use global step instead of local step
                    "n": self.episode_count
                }
                
                # Log individual episode directly if logger is available
                if self.logger is not None:
                    # Log overall episode metrics
                    self.logger.log_episode_info(
                        episode_num=self.episode_count,
                        reward=episode_reward,
                        length=episode_length,
                        global_step=self.global_step  # Use global step instead of local step
                    )
                
                # Reset the episode tracking for this environment
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
            infos.append(info_dict)

        return obs, rewards, dones, infos

    def close(self, **kwargs):
        return self.env.close()

    @property
    def use_monitor(self):
        return False

    def batch_rewards(self, buffer):
        step_count = self.env.reward_buffer["step count"]
        reward_dict = {"average step reward": 0}
        
        # Process step rewards as before
        for k in self.env.reward_buffer.keys():
            if k == "step count":
                continue
            reward_dict[k] = self.env.reward_buffer[k] / (self.num_envs * step_count)
            if hasattr(self.env, "single_agent_reward_scale"):
                reward_dict[k] *= self.env.single_agent_reward_scale
            if "reward" in k or "punishment" in k:
                reward_dict["average step reward"] += reward_dict[k]
            self.env.reward_buffer[k] = 0
        self.env.reward_buffer["step count"] = 0
        
        return reward_dict

    def log_episode_rewards(self):
        # Log episode rewards
        print(f"Episode {self.episode_count}: reward={self.episode_rewards.mean().item():.2f}, length={self.episode_lengths.mean().item():.2f}")

class MATWrapper(gym.Wrapper):
    @property
    def observation_space(
        self,
    ):
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            observation_space = self.env.observation_space
        else:
            observation_space = self._observation_space

        if (
            "critic" in observation_space.spaces.keys()
            and "policy" in observation_space.spaces.keys()
        ):
            observation_space = observation_space["policy"]
        return observation_space

    def observation(self, observation):
        if self._observation_space is None:
            observation_space = self.env.observation_space
        else:
            observation_space = self._observation_space

        if (
            "critic" in observation_space.spaces.keys()
            and "policy" in observation_space.spaces.keys()
        ):
            observation = observation["policy"]
        return observation

    def reset(self, **kwargs):
        """Reset all environments."""
        return self.env.reset(**kwargs)

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        return self.env.step(actions, extra_data)

class SingleAgentWrapper(gym.Wrapper):

    def __init__(self, env):
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)

        self.num_envs = self.env.num_envs * self.env.num_agents
        self.num_agents = 1

        self.single_agent_reward_scale = self.env.num_agents

    def reset(self, **kwargs):
        """Reset all environments."""
        obs = self.env.reset(**kwargs)
        return obs.reshape(self.num_envs, 1, -1)

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        actions = actions.reshape(self.env.num_envs, self.env.num_agents, -1)
        obs, reward, termination, info = self.env.step(actions)
        return obs.reshape(self.num_envs, 1, -1), reward.reshape(self.num_envs, 1), torch.stack([termination, termination], dim=1).reshape(self.num_envs), info

def parse_arguments(parser, headless=False, no_graphics=False, custom_parameters=[]):

    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args

def get_args():

    openrl_parser = create_config_parser()

    custom_parameters = [
        {"name": "--task", "type": str, "default": "go1gate", "help": "Select task via name"},
        {"name": "--algo", "type": str, "default": "ppo", "help": "Select pipeline via name"},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str,  "help": "Saved model checkpoint path. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--train_timesteps", "type": int, "help": "Maximum number of training time steps. Overrides config file if provided."},

        {"name": "--use_wandb", "action": "store_true", "default": False, "help": "Use wandb for record"},
        {"name": "--use_tensorboard", "action": "store_true", "default": False, "help": "Use tensorboard for record"},
        {"name": "--exp_name", "type": str, "default": "default"},
        {"name": "--record_video", "action": "store_true", "default": False},
        {"name": "--num_episodes", "type": int, "default": 5, "help": "Number of episodes to run during evaluation"},
    ]
    # parse arguments
    args = parse_arguments(
        openrl_parser,
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    
    # Set num_env_steps to match train_timesteps to fix driver error
    if hasattr(args, 'train_timesteps') and args.train_timesteps is not None:
        args.num_env_steps = args.train_timesteps
    else:
        # Default value if train_timesteps is not provided
        args.num_env_steps = 1000000
        
    return args
