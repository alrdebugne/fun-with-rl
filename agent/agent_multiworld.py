import logging
from pathlib import Path
import pickle
import random
from typing import Optional, Tuple, Union

import math
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .agent import DDQNAgent
from .solver import DQNetwork


class MultiworldDDQNAgent(DDQNAgent):
    """
    Builds on top of DDQNAgent to allow learning across multiple environments at once.
    The memory buffer is apportioned in N chunks, each dedicated to one env at a time,
    to avoid catastrophic forgetting (with N is the number of environments).

    TODO: make sampling of indices in `recall` robust.
    Currently, recall samples from range(self.memory_num_experiences). This can point to
    empty indices if we switch to another env before its apportioned chunk of memory is
    filled up.
    """

    def __init__(
        self,
        n_envs: int,
        *super_args,
        **super_kwargs,
    ):
        super(MultiworldDDQNAgent, self).__init__(*super_args, **super_kwargs)
        self.n_envs = n_envs
        # Create helper memory pointers to apportion the memory in equal chunks for every env
        self.memory_pointer_per_env = {i: 0 for i in range(n_envs)}
        self.max_memory_size_per_env = self.max_memory_size / n_envs
        if not self.max_memory_size_per_env.is_integer():
            self.max_memory_size_per_env = int(math.floor(self.max_memory_size_per_env))
            logging.warning(
                f"Memory size {self.max_memory_size} cannot be divided in {n_envs} equal chunks. "
                f"Setting memory size to {self.max_memory_size_per_env} per env instead."
            )
    
    # Getter
    def get_env_index(self) -> int:
        return self._env_index

    # Setter
    def set_env_index(self, val: int) -> None:
        self._env_index = int(val)

    # Property
    env_index = property(get_env_index, set_env_index)

    def update_memory_pointer_and_count(self) -> None:
        """
        Updates index for memory pointer and number of experiences in memory, apportioning
        a contiguous buffer of size `max_memory_size_per_env` for each env.
        Must be called _before_ each call to `remember`.

        TODO: figure out how to pass env_index without copying large parts
        of super().play_episde()?
        """
        # Update position of pointer inside chunk dedicated to env
        env_index = self.env_index
        # FIXME: ^ set outside agent in run loop...
        self.memory_pointer_per_env[env_index] = (
            self.memory_pointer_per_env[env_index] + 1
        ) % self.max_memory_size_per_env
        # Update global pointer, taking into account starting position
        self.memory_pointer = int(
            env_index * self.max_memory_size_per_env
            + self.memory_pointer_per_env[env_index]
        )
        self.memory_num_experiences = min(
            self.memory_num_experiences + 1, self.max_memory_size
        )
