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

        # Set up twin networks (same architecture as Mnih et al. 2013)
        if not from_pretrained:
            self.q1 = CategoricalCNN(state_space, action_space)
            self.q2 = CategoricalCNN(state_space, action_space)
        else:
            self.q1 = torch.load(save_dir / Path("q1.pt"), map_location=torch.device(self.device))
            self.q2 = torch.load(save_dir / Path("q2.pt"), map_location=torch.device(self.device))

        # Optimisation set up outside the agent
        # self.optimiser = torch.optim.Adam(self.parameters(), lr=3e-4)
        # self.loss = nn.SmoothL1Loss(beta=1.0).to(self.device)
        # self.epsilon = 0.1
        # # TODO: if resuming training, copy learning rate from previous optimiser

    def act(self, s: torch.Tensor) -> torch.Tensor:
        """ Pick next action following epsilon-greedy policy """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            # Q-clipping
            return torch.argmax(min(self.q1(s), self.q2(s)))
        # TODO: check how to return tensor 'nested right'