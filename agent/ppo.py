import logging
import torch
from torch.distributions import Categorical
from typing import Dict, Tuple

from agent.core import ActorCritic

logger = logging.getLogger("agent-a2c-ppo")


class A2CPPO(ActorCritic):
    """ """

    def __init__(self, *super_args, **super_kwargs):
        super(A2CPPO, self).__init__(*super_args, **super_kwargs)

    def update(self, data: dict) -> None:  # type: ignore
        """
        Updates policy & value function networks using PPO. In the parent class ActorCritic,
        these networks are defined as self.pi and self.vf, respectively.

        Inputs:
            data: experience buffer storing variables required for learning;
            see buffers/ppo.py:PPOBuffer. Must contain the following:
                observations s(t)
                actions a(t)
                returns R(t)
                advantages A(t)
                log-probs log[pi(a_t|s_t)] (from pi used for sampling)
        """
        raise NotImplementedError

    def _compute_loss_pi(
        self, data: Dict[str, torch.Tensor], epsilon: float = 0.2
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute policy loss under PPO, i.e. as a clipped trust-region PO:
            l_ppo_t = min[ r_t * adv_t,  clip( r_t, 1 - epsilon, 1 + epsilon ) * adv_t ],
        where r_t = pi_{theta_new}(a_t|s_t) / pi_{theta_old}(a_t|s_t), and
            loss_ppo = -mean(l_ppo_t)

        Here, `data` is a dict storing statistics for the past epoch.
        See buffers/ppo.py:PPOBuffer for details.
        """

        states = data["obs"]
        actions = data["act"]
        advantages = data["adv"]
        logp_a_old = data["logp_a"]  # based on old policy pi

        # Compute logp_a under current policy pi
        pi_new = Categorical(logits=self.pi(states))
        logp_a_new = pi_new.log_prob(actions)
        # Compute policy loss under PPO
        ratio = torch.exp(logp_a_new - logp_a_old)  # r_t
        ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        pi_loss = -(torch.min(ratio * advantages, ratio_clipped * advantages)).mean()

        # Extract useful statistics: magnitude of policy update, new policy entropy
        kl_approx = (logp_a_old - logp_a_new).mean().item()
        entropy = pi_new.entropy().mean().item()
        return pi_loss, {"kl_div": kl_approx, "entropy": entropy}

    def compute_loss_vf(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss of value function V(s_t) as MSE of (observed) returns and
        values predicted by the critic.

        Here, `data` is a dict storing statistics for the past epoch.
        See buffers/ppo.py:PPOBuffer for details.
        """
        states = data["obs"]
        observed = data["ret"]
        predicted = self.vf(states).squeeze()
        return ((predicted - observed) ** 2).mean()
