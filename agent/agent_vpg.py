import logging
from pathlib import Path
import pickle
import random
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .solver import DQNetwork
from utils import render_in_jupyter

logger = logging.getLogger("vpg-agent")


class VPGAgent:
    """
    Agent class, using Vanilla Policy Gradient with GAE-lambda advantage estimation

    Main methods:
    - XXX

    Current simplifications:
    - Rewards-to-go used insted of advantage function inside log-grad policy
    - Rewards are undiscounted
    """

    def __init__(
        self,
        state_space: npt.NDArray[np.float64],
        action_space: int,
        lr: float,  # alpha for gradient ascent
        dropout: float,
        exploration_max: float,
        exploration_min: float,
        exploration_decay: float,
        save_dir: Path = Path("./data/tmp/"),
        is_pretrained: bool = False,
    ):
        """ """
        # ~~~ Define layers for policy network ~~~
        self.state_space = state_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        self.is_pretrained = is_pretrained

        # Define network for policy
        self.policy = DQNetwork(state_space, action_space, dropout).to(self.device)
        if self.is_pretrained:
            if self.save_dir is None:
                raise ValueError("`save_dir` must be specified for resuming training")
            logger.info("Loading weights from previous runs...")
            # Load weights from previous iteration
            self.policy.load_state_dict(
                torch.load(
                    self.save_dir / Path("vpg_gae.pt"),
                    map_location=torch.device(self.device),
                )
            )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.step = 0  # initialise steps

        # ~~~ Learning parameters ~~~
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_min
        # ^ temporary, before implementing update_exploration_rate

    def act(self, state) -> torch.Tensor:
        """
        Epsilon-greedy policy: output next action given `state`, either by:
        - exploring randomly, with proba. `self.exploration_rate`
        - picking best action from policy
        """
        self.step += 1
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        return (
            torch.argmax(self.policy(state.to(self.device)))
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
            # ^ calling unsqueeze to return 'nested' tensor
        )

    def play_episode(
        self, env, render: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.float64], int]:
        """
        Plays one episode from start to finish in `env` (i.e. one trajectory).
        Returns the associated onservations, actions & rewards (for gradient descent)
        & steps (for logging).
        """

        state = env.reset()
        state = torch.Tensor(np.array([state]))
        observations = []
        actions = []
        rewards = []
        steps_episode = 0
        done = False

        # Render?
        if render:
            # Initialise variables required for rendering
            img = None
            reward = 0

        start = time.time()
        while not done:
            if render:
                img = render_in_jupyter(env, img, info=f"Current reward: {reward}")

            # Store current state
            observations.append(state)
            # Pick next action and step environment forward
            action = self.act(state)
            actions.append(action)
            steps_episode += 1
            state_next, reward, done, info = env.step(int(action[0]))

            # Format to pytorch tensors
            state_next = torch.Tensor(np.array([state_next]))
            rewards.append(reward)
            state = state_next

        end = time.time()

        logger.debug(
            f"--- Computed trajectory in {int(end - start):.2f} s (reward: {sum(rewards)}, steps: {steps_episode})"
        )

        return observations, actions, rewards, steps_episode

    def play_epoch(
        self, env, steps_per_epoch: int, render: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Samples trajectories inside env using the latest policy until a batch of size at least
        `steps_per_epoch` is assembled.
        Returns observations
        """

        steps = 0
        observations = []
        actions = []
        weights = []

        has_rendered_epoch = False

        while steps < steps_per_epoch:
            if render and not has_rendered_epoch:
                render_episode = True
                has_rendered_epoch = True
            else:
                render_episode = False

            (
                obs_episode,
                actions_episode,
                rewards_episode,
                steps_episode,
            ) = self.play_episode(env, render=render_episode)

            observations.extend(obs_episode)
            actions.extend(actions_episode)
            # Compute weights from rewards
            weights_episode = self.compute_weights(rewards_episode)
            weights.extend(weights_episode)
            steps += steps_episode

        return observations, actions, weights

    def compute_weights(self, rewards_trajectory: List[np.float64]) -> Any:
        """
        Computes weights for every stage transition t in the trajectory.
        In the simplest formulations, weights are the rewards-to-go for the trajector.
        More complex formulations are found in the children classes of VPGActor.
        """

        # `rewards_trajectory` contains the reward r(t) collected at every time step
        # We want to compute the sum of (undiscounted) rewards that each action enabled,
        # not including past rewards; i.e. we want
        #   w(t) = sum_{t'=t}^{T} r(t')

        n = len(rewards_trajectory)
        weights_trajectory = np.zeros_like(rewards_trajectory)
        for i in reversed(range(n)):
            weights_trajectory[i] = rewards_trajectory[i] + (
                weights_trajectory[i + 1] if i + 1 < n else 0
            )
            # ^ faster than looping forward with weights[i] = rewards[i:].sum()

        return weights_trajectory

    def compute_loss(
        self, states: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor
    ) -> Any:
        """
        Computes 'loss' for a batch of states.

        N.B.: not really a 'loss' in the sense of performance metric, but an objective
        that we train the policy network to maximise.

        Params:
            states: list of observations
            actions: list of actions taken (their index)
            weights: weights by which to multiply the log(policy) terms; exact formulation
                     varies, but is returned by self.compute_weights()
        """
        m = Categorical(logits=self.policy(states))  # m for multinomial
        log_policy = m.log_prob(actions)
        # Breaking the steps down for torch newbies:
        # - `policy` returns the logits over the action space for a given state.
        #    For a single state, `policy` returns a tensor of size (`self.action_space`).
        #    For a N states, `policy` returns a tensor of size (N, `self.action_space`).
        # - torch.distribution.Categorical converts logits to a (multinomial) distribution object.
        #   This object has useful methods, such as .log_prob(). Its size is that of the logits.
        # - Calling .log_prob(actions) evaluates log( policy(a|s) ) for every pair (a, s)
        #   in (actions, states).
        # - The last step is to weigh the logs by the weights (e.g. rewards-to-go, advantage), then
        #   compute the batch average loss with .mean(). We add a minus sign because we're computing
        #   the total value of trajectory (the greater, the better), not a loss.
        return -(log_policy * weights).mean()

    def save(self, dir: Path) -> None:
        """Saves memory buffer and network parameters"""

        dir.mkdir(parents=True, exist_ok=True)

        # Save rewards accumulated at each epoch
        # with open(dir / Path("rewards.pkl"), "wb") as f:
        #     pickle.dump(self.rewards, f)
        torch.save(self.policy.state_dict(), dir / Path("vpg_gae.pt"))

    def run(
        self,
        env,
        num_epochs: int,
        batch_size: int,
        save_after_epochs: int,
        print_progress_after: int = 50,
    ) -> None:
        """
        Updates policy over the course of `num_epoch` epochs of size `batch_size`,
        saving progress every `save_after_epochs` steps.

        Recommended calls wrap `env` to speed up learning, e.g.:

        ```python
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
        env = make_env(env)  # wrap
        agent = VPGAgent(...)
        agent.run(env, num_epochs=1000, save_after_epochs=1000)
        ```

        TODO: the code is inefficient right now because it uses two forward loops instead of one:
        one when calling self.act() to get the action, then another when computing losses (on a batch).
        Another consequence of that is that we need to store observations, because we need them for computing
        losses, too. Refactor this.
        """

        if not isinstance(self.save_dir, Path):
            e = f"Called method run(), but agent has invalid save dir: {self.save_dir}"
            raise ValueError(e)

        logger.info(
            f"Starting training for {num_epochs} epochs with parameters: ... (TODO)"
        )
        start = time.time()

        for epoch in range(num_epochs):
            # Sample observations and rewards from one epoch
            batch_observations, batch_actions, batch_weights = self.play_epoch(
                env, batch_size
            )
            # self.update_exploration_rate()  # TODO
            # Perform policy upgrade step
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch_observations, batch_actions, batch_weights)
            loss.backward()
            self.optimizer.step()

            if (epoch > 0) & (epoch % print_progress_after == 0):
                logger.info(f"Loss after epoch {epoch}: {loss:.2f}")

            if (epoch > 0) & (epoch % save_after_epochs == 0):
                logger.info(f"Saving progress at episode {epoch}...")
                self.save(dir=self.save_dir)
                logger.info("Done.")

        end = time.time()
        logger.info(f"Run complete (runtime: {round(end - start):d} s)")
        logger.info(f"Final loss: {loss}")
        logger.info("Saving final state...")
        self.save(dir=self.save_dir)
        logger.info("Done.")
