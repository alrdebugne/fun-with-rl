import logging
from pathlib import Path
from typing import *

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import random

from utils import get_n_trainable_params

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ddqn-agent")

class DDQNAgent(nn.Module):
    """
    Agent class for implementing the Double Deep Q-Networks algorithm.
    Depending on the loss function that's used, this class can be used
    to implement either DDQN or TD3 (for discrete actions).
    
    Methods:
        `act`: picks an action based on epsilon-greedy policy
        `save`: saves network weights to disc
    """
    def __init__(
        self,
        state_space: npt.NDArray[np.float64],
        action_space: int,
        network_class: nn.Module,
        network_kwargs: dict,
        from_pretrained: bool,
        save_dir: Union[Path, str] = Path("models"),
    ) -> None:
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epsilon = 0.10 # TODO: warm-up steps + decay

        # Set up twin networks (same architecture as Mnih et al. 2013)
        self.save_dir = save_dir
        self.q1 = network_class(**network_kwargs).to(self.device) # primary
        self.q2 = network_class(**network_kwargs).to(self.device) # target
    
        if from_pretrained:
            logger.info("Loading pretrained weights from {save_dir}")
            self.q1.load_state_dict(
                torch.load(save_dir / Path("q1.pt"), map_location=torch.device(self.device))
            )
            self.q2.load_state_dict(
                torch.load(save_dir / Path("q2.pt"), map_location=torch.device(self.device))
            )

        n_params = get_n_trainable_params(self.q1)
        logger.info(f"Created agent with {n_params} parameters (online Q)")

        # Optimisation set up outside the agent


    def act(self, s: torch.Tensor) -> int:
        """ Pick next action following epsilon-greedy policy """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                # DDQN
                return torch.argmax(self.q1(s)).item()
                # TD3 (Q-clipping)
                # return torch.argmax(torch.min(self.q1(s), self.q2(s))).item()
    
    def save(self) -> None:
        """Saves network parameters"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Save model weights
        torch.save(self.q1.state_dict(), self.save_dir / Path("q1.pt"))
        torch.save(self.q2.state_dict(), self.save_dir / Path("q2.pt"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for training agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_loss_ddqn(
    agent: nn.Module,
    transitions: Dict[str, torch.Tensor],
    loss_fn: Callable,
    gamma: float,
    return_errors: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes loss using transitions from replay buffer
        predictions = Q1(s)|a (primary network)
        targets = r + (1-d) * gamma * Q2(s') (target network)
    
    If `return_errors`, also returns the TD(0) error (in absolute value).
    This is mostly used for prioritised experience replay.
    """
    s = transitions["s"]
    a = transitions["a"].view(-1, 1).long()
    # ^ adds new dimension for broadcasting
    r = transitions["r"]
    s_next = transitions["s_next"]
    d = transitions["d"]

    # Compute predictions & target
    # I'm confused as to which procedure to follow... From what I can tell,
    # there are two approaches to DDQN.

    # DDQN as described in Hasselt et al. 2016:
    # - online network computes Q-value for s_t and picks a_{t+1}
    # - target network evaluates Q-value for s_{t+1}, a_{t+1}
    preds = agent.q1(s).gather(1, a).squeeze()
    a_next = agent.q1(s_next).max(1)[1].view(-1, 1) # [0] is values, [1] is indices
    targets = r + (1 - d) * gamma * agent.q2(s_next).gather(1, a_next).squeeze()

    # DDQN as described in Mnih et al. 2015:
    # - online network computes Q-value for s_t
    # - target network picks a_{t+1} and evaluates Q-values for s_{t+1}, a_{t+1}
    # ¯\_(ツ)_/¯
    # preds = agent.q1(s).gather(1, a).squeeze()
    # targets = r + (1 - d) * gamma * agent.q2(s_next).max(1)[0]

    if return_errors:
        return loss_fn(preds, targets), torch.abs(preds - targets).detach()
    return loss_fn(preds, targets)


def compute_loss_td3(agent: nn.Module, transitions: Dict[str, torch.Tensor], loss_fn: Callable, gamma: float) -> torch.Tensor:
    """
    Computes loss using transitions from replay buffer
        predictions = Q1(s)|a, Q2(s)|a (i.e. Q-functions evaluated at (s,a))
        targets = r + (1-d) * gamma * (min(Q1(s'), Q2(s'))
    """
    s = transitions["s"]
    a = transitions["a"].view(-1, 1).long()
    # ^ adds new dimension for broadcasting
    r = transitions["r"]
    s_next = transitions["s_next"]
    d = transitions["d"]

    # Compute predictions & target
    # Predictions
    q1_preds = agent.q1(s).gather(1, a).squeeze()
    q2_preds = agent.q2(s).gather(1, a).squeeze()
    # Targets
    q_clipped = torch.max(torch.min(agent.q1(s_next), agent.q2(s_next)), 1)[0]
    # ^ torch.max() with dim returns values and indices; [0] accesses values
    targets = r + (1 - d) * gamma * q_clipped
    # Losses
    loss_q1 = loss_fn(q1_preds, targets)
    loss_q2 = loss_fn(q2_preds, targets)

    return loss_q1 + loss_q2
