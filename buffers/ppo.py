import logging


class PPOBuffer:
    """
    Stores trajectories for a PPO agent interacting with its environment (using GAE),
    and provides methods for retrieving interactions from buffer (e.g. for learning).

    Inspired from OpenAI's implementation in spinning-up.
    """

    def __init__(self, gamma: float, _lambda: float) -> None:
        self.pointer = 0
        self.gamma = gamma
        self._lambda = _lambda

    def store(self) -> None:
        """Appends one timestep of agent-env. interaction to the buffer"""
        self.pointer += 1

    def finish_path(self) -> None:
        """ """
        raise NotImplementedError

    def get(self) -> None:
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
