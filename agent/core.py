import abc
import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import Categorical
import torch.nn as nn

from agent import CategoricalCNN, CategoricalMLP

logger = logging.getLogger("agent-core")


class ActorCritic(nn.Module):
    """
    Core actor-critic class for agents interacting with an environment.
    Works for categorical outputs only.

    Methods:
        `step`: steps through environment once and returns action, est. value & log probs
        `act`: steps through environment once and returns action
        `update`: performs update on the policy and value networks

    Notes:
        `forward` (required for torch Modules) is defined on the networks `pi` and `vf`
    """

    def __init__(
        self,
        # ~ Env. variables ~
        state_space: npt.NDArray[np.float64],
        action_space: int,
        # ~ Actor variables ~
        policy_lr: float,
        # ~ Critic variables ~
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

        # Define type of network that's required based on the input data
        # Only two cases are supported so far:
        # * Input = frame(s): use CategoricalCNN
        # * Input = 1D-array(s): use CategoricalMLP
        network_class: nn.Module
        if len(state_space) == 3:
            network_class = CategoricalCNN
        elif len(state_space) == 1:
            network_class = CategoricalMLP
        else:
            e = (
                f"Only `state_space`s of dimensions 1 (1D-arrays) or 3 (stack of frames) "
                f"are supported, but got state_space {state_space}"
            )
            raise AssertionError(e)

        # ~~~ Actor: define policy network ~~~
        self.pi = network_class(input_shape=state_space, n_actions=action_space)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=policy_lr)

        # ~~~ Critic: define value function network ~~~
        self.vf = network_class(input_shape=state_space, n_actions=1)
        # ^ `n_actions = 1` returns a float (suitable for value func. approx.)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=value_func_lr)

        # Log number of variables
        num_variables = tuple(
            sum([np.prod(p.shape) for p in module.parameters()])
            for module in [self.pi, self.vf]
        )
        logger.info("Number of parameters: \t pi: %d, \t V: %d" % num_variables)

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

    def step(
        self, states: torch.Tensor
    ) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        """
        Performs one step from states s_t and returns the actions, estimated values
        and log prob. of those actions.

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


def combine_shape(n: int, shape: Union[int, List, Tuple, None] = None) -> Tuple:
    """Small utility function to combine n samples of dimension `shape` into a tuple"""
    if shape is None:
        return (n,)
    return (n, shape) if np.isscalar(shape) else (n, *shape)  # type: ignore


def discounted_cumsum(
    array: npt.NDArray[np.float32], discount: float
) -> npt.NDArray[np.float32]:
    """
    Computes cumulative sum of `array` discounted by `discount`.
    To be used for computing rewards-to-go and GAE-lambda.

    Ex.:
        Inputs:
            array = [x, y, z]; discount = gamma
        Returns:
            [x + gamma * y + gamma ** 2 * z,
             y + gamma * z,
             z]

    """
    n = len(array)
    cumsums = np.zeros_like(array)
    for i in reversed(range(n)):
        cumsums[i] = array[i] + discount * (cumsums[i + 1] if i + 1 < n else 0)
    return cumsums
