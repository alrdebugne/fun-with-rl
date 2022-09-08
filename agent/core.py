import abc
import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import Categorical
import torch.nn as nn

from agent import FrameToActionNetwork

logger = logging.getLogger("agent-core")


class ActorCritic(nn.Module):
    """
    Core actor-critic class for agents interacting with an environment.
    Works for categorical outputs only.

    Methods:
        `step`
        `act`
        `update`

    Notes:
        `forward` (required for torch Modules) is defined on the networks `pi` and `vf`
    """

    def __init__(
        self,
        # ~ Env. variables ~
        state_space: npt.NDArray[np.float64],
        action_space: int,
        # ~ Actor variables ~
        gamma: float,
        policy_lr: float,
        # ~ Critic variables ~
        _lambda: float,
        value_func_lr: float,
        # ~ Where to save / read savestates ~
        save_dir: Union[Path, None] = None,
        is_pretrained: bool = False,
    ):
        super(ActorCritic, self).__init__()
        # ^ see https://pytorch.org/docs/stable/generated/torch.nn.Module.html

        self.state_space = state_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        self.is_pretrained = is_pretrained

        # ~~~ Actor: define policy network ~~~
        self.pi = FrameToActionNetwork(input_shape=state_space, n_actions=action_space)
        self.gamma = gamma
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=policy_lr)

        # ~~~ Critic: define value function network ~~~
        self.vf = FrameToActionNetwork(input_shape=state_space, n_actions=1)
        # ^ `n_actions = 1` returns a float (suitable for value func. approx.)
        self._lambda = _lambda
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=value_func_lr)

        # ~~~ Loading from previous runs (if applies) ~~~
        if self.is_pretrained:
            logger.info("Loading weights from previous runs...")
            # Load weights from previous iteration
            self.pi.load_state_dict(
                torch.load(
                    self.save_dir / Path("pi.pt"),  # type: ignore
                    map_location=torch.device(self.device),
                )
            )
            self.vf.load_state_dict(
                torch.load(
                    self.save_dir / Path("vf.pt"),  # type: ignore
                    map_location=torch.device(self.device),
                )
            )

    # fmt: off
    def step(
        self, states: torch.Tensor
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # fmt: on
        """
        Performs one step in current state s_t and returns the action, estimated value
        and log prob. of returned action.

        Notes:
            `states` must be a stack of frames, i.e. of dimension (n, *state_space). If `states` is a single
            state instead, call states.unsqueeze(0).
        """
        with torch.no_grad():  # speeds up calcs to explicitly not calculate gradients
            logits = self.pi(states)
            pi = Categorical(logits=logits)
            a = pi.sample()
            v = self.vf(states)
            log_prob_a = pi.log_prob(a)
        # TODO: check if needs to call to self.device() on state, a
        return a.numpy(), v.numpy(), log_prob_a.numpy()

    def act(self, states) -> npt.NDArray[np.float32]:
        """Samples next action from policy pi(a_t|s_t)"""
        return self.step(states)[0]

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:  # type: ignore
        """
        Call to update policy and value func. networks.
        Must be implemented by subclasses (e.g. PPOActorCritic).
        """
