import logging

from agent.core import ActorCritic

logger = logging.getLogger("agent-a2c-ppo")


class A2CPPO(ActorCritic):
    """ """

    def __init__(self, *super_args, **super_kwargs):
        super(A2CPPO, self).__init__(*super_args, **super_kwargs)

    def update(self, buffer: dict) -> None:  # type: ignore
        """
        Updates policy & value function networks using PPO. In the parent class ActorCritic,
        these networks are defined as self.pi and self.vf, respectively.

        Inputs:
            buffer: experience buffer storing variables required for learning.
            Must contain the following:
                observations s(t)
                actions a(t)
                returns R(t)
                advantages A(t)
                log-probs log[pi(a_t|s_t)] (from pi used for sampling)
        """
        raise NotImplementedError
