import numpy as np
import numpy.typing as npt
import logging
from typing import List, Tuple, Union


def combine_shape(n: int, shape: Union[int, List, Tuple, None] = None):
    """Small utility function to combine n samples of dimension `shape` into a tuple"""
    if shape is None:
        return (n,)
    return (n, shape) if np.isscalar(shape) else (n, *shape)  # type: ignore


def compute_rewards_to_go() -> None:
    pass


class PPOBuffer:
    """
    Stores trajectories for an agent interacting with its environment (using GAE),
    and provides methods for retrieving interactions from buffer (e.g. for learning).

    Inspired from OpenAI's implementation in spinning-up.
    """

    def __init__(
        self,
        state_space: npt.NDArray[np.float64],
        action_space: int,
        size: int,
        gamma: float,
        _lambda: float,
    ) -> None:

        self.gamma = gamma
        self._lambda = _lambda
        self.size = size

        # ~~~ Initialise buffers ~~~
        self.observations = np.zeros(combine_shape(size, state_space), dtype=np.float32)
        self.actions = np.zeros(combine_shape(size, action_space), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.rewards_to_go = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.est_values = np.zeros(size, dtype=np.float32)
        self.logp_actions = np.zeros(size, dtype=np.float32)

        # ~~~ Initialise pointers ~~~
        self.pointer = 0  # current pointer in buffer
        self.path_start_index = (
            0  # pointer for start of current path (needed for termination)
        )

    def store(self, obs, act, rew, val, logp) -> None:
        """Appends one timestep of agent-env. interaction to the buffer"""
        assert self.pointer < self.size
        self.observations[self.pointer] = obs
        self.actions[self.pointer] = act
        self.rewards[self.pointer] = rew
        self.est_values[self.pointer] = val
        self.logp_actions[self.pointer] = logp
        # ^ rewards-to-go and advantages computed & stored after path finishes

        # Increment pointer
        self.pointer += 1

    def finish_path(self) -> None:
        """ """
        raise NotImplementedError

    def get(self) -> dict:
        """
        Returns batch of agent-env. interactions from the buffer (for learning) & resets pointer
        for next trajectory.

        Returns:
            observations
            actions
            rewards-to-go
            advantages
            action log-probs, i.e. log[pi(a_t|s_t)]
        """
        self.pointer = 0
        raise NotImplementedError
