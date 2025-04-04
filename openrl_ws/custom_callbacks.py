"""
Custom callbacks for OpenRL that properly handle evaluation environments
"""

import os
import numpy as np
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

from openrl.utils.callbacks.eval_callback import EvalCallback as OriginalEvalCallback
from openrl.utils.callbacks import BaseCallback, EventCallback
from openrl.utils.callbacks import callbacks_factory
from openrl.envs.vec_env import BaseVecEnv, SyncVectorEnv
from openrl.utils.evaluation import evaluate_policy
from openrl.utils.callbacks.callbacks_factory import CallbackFactory

import gymnasium as gym


class CustomEvalCallback(OriginalEvalCallback):
    """
    Modified version of EvalCallback that properly handles the case when eval_env is None.
    In this case, it will use a copy of the training environment for evaluation.
    """

    def __init__(
        self,
        eval_env=None,
        callbacks_on_new_best=None,
        callbacks_after_eval=None,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        render=False,
        asynchronous=True,
        verbose=1,
        warn=True,
        stop_logic="OR",
        close_env_at_end=True,
    ):
        # Initialize with a placeholder environment if eval_env is None
        # The actual environment will be set in _init_callback
        super(EventCallback, self).__init__(callbacks_after_eval, verbose=verbose)
        self.stop_logic = stop_logic
        
        if isinstance(callbacks_on_new_best, list):
            callbacks_on_new_best = callbacks_factory.CallbackFactory.get_callbacks(
                callbacks_on_new_best, stop_logic=stop_logic
            )

        self.callbacks_on_new_best = callbacks_on_new_best

        if self.callbacks_on_new_best is not None:
            # Give access to the parent
            self.callbacks_on_new_best.set_parent(self)

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.close_env_at_end = close_env_at_end
        
        # Store eval_env for later use in _init_callback
        self._eval_env_config = eval_env
        self.eval_env = None  # Will be set in _init_callback
        
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_time_steps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Create the eval environment if it doesn't exist
        if self.eval_env is None:
            # Use a copy of the training environment
            self.eval_env = self.training_env
            
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callbacks_on_new_best is not None:
            self.callbacks_on_new_best.init_callback(self.agent)


# Register the custom callback
CallbackFactory.register("CustomEvalCallback", CustomEvalCallback)
