import logging
from pathlib import Path
from typing import *

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import random

from .solver import CategoricalCNN

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ddqn-agent")

class DDQNAgent(nn.Module):
    """
    Agent class for implementing the Double Deep Q-Networks algorithm,
    with Q-clipping borrowed from TD3.
    
    Methods:
        `act`: picks an action based on epsilon-greedy policy
    """
    def __init__(
        self,
        state_space: npt.NDArray[np.float64],
        action_space: int,
        from_pretrained: bool,
        save_dir: Union[Path, str] = Path("models"),
    ) -> None:
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epsilon = 0.10 # TODO: warm-up steps

        # Set up twin networks (same architecture as Mnih et al. 2013)
        self.save_dir = save_dir
        if not from_pretrained:
            self.q1 = CategoricalCNN(state_space, action_space).to(self.device)
            self.q2 = CategoricalCNN(state_space, action_space).to(self.device)
        else:
            self.q1 = torch.load(save_dir / Path("q1.pt"), map_location=torch.device(self.device))
            self.q2 = torch.load(save_dir / Path("q2.pt"), map_location=torch.device(self.device))

        # Optimisation set up outside the agent

    def act(self, s: torch.Tensor) -> int:
        """ Pick next action following epsilon-greedy policy """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            # Q-clipping
            with torch.no_grad():
                return torch.argmax(torch.min(self.q1(s), self.q2(s))).item()
        # TODO: check how to return tensor 'nested right'
    
    def save(self) -> None:
        """Saves network parameters"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Save model weights
        torch.save(self.q1.state_dict(), self.save_dir / Path("q1.pt"))
        torch.save(self.q2.state_dict(), self.save_dir / Path("q2.pt"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for training agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_loss(agent: nn.Module, transitions: Dict[str, torch.Tensor], loss_fn: Callable, gamma: float) -> torch.Tensor:
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