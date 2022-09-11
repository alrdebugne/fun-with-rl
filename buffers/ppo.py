import numpy as np
import numpy.typing as npt

from agent.core import combine_shape, discounted_cumsum


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

    def finish_path(self, last_value: float = 0.0) -> None:
        """
        To be called at the end of a trajectory, or when a trajectory is terminated early
        (e.g. when an epoch ends, or during timeout).

        Computes rewards-to-go and (normalised) advantages for the trajectory (using GAE-lambda).
        When a trajectory is cut off early, uses the estimated value function for the last state,
        `last_value` V(s_T), to bootstrap the rewards-to-go calculation.
        """
        path_slice = slice(self.path_start_index, self.pointer)
        path_rewards = self.rewards[path_slice]
        path_values = self.est_values[path_slice]

        # TODO:
        # 1. Calculate discounted rewards-to-go
        # 2. Calculate GAE-lambda

        # Update index for start of next path
        self.path_start_index = self.pointer

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
