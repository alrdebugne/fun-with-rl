import logging
from pathlib import Path
import time
import typing
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils import render_in_jupyter

logger = logging.getLogger("vpg-agent")


class VPGAgent:
    """
    Agent class, using Vanilla Policy Gradient with rewards-to-go
    """

    def __init__(
        self,
        state_space: npt.NDArray[np.float64],
        action_space: int,
        policy_net: nn.Module,
        policy_net_kwargs: dict,
        gamma: float,
        lr: float,  # alpha for gradient ascent
        save_dir: Union[Path, None] = None,
        save_name: Union[Path, None] = None,
        is_pretrained: bool = False,
    ):
        """ """
        # ~~~ Define layers for policy network ~~~
        self.state_space = state_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        self.save_name = save_name
        self.is_pretrained = is_pretrained

        # Define network for policy
        self.policy = policy_net(**policy_net_kwargs).to(self.device)
        if self.is_pretrained:
            if self.save_dir is None or self.save_name is None:
                raise ValueError("`save_dir` must be specified for resuming training")
            logger.info("Loading weights from previous runs...")
            # Load weights from previous iteration
            self.policy.load_state_dict(
                torch.load(
                    self.save_dir / self.save_name,  # type: ignore
                    map_location=torch.device(self.device),
                )
            )
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.step = 0  # initialise steps

    def play_episode(
        self, env, render: bool = False, exploit_only: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.float64], int]:
        """
        Plays one episode from start to finish in `env`.
        For learning, `exploit_only` must be set to False. Set `exploit_only`
        to True to follow the optimal policy.

        Returns:
            observations: list of states observed at t
            actions: list of actions taken at t
            rewards: list of rewards earned at time t
            steps: length of episode (for logging)

        Note the returns have different types than `play_epoch`.
        """

        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32)
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
            # Pick next action
            action = self._act(state.unsqueeze(0), exploit_only)
            # Step environment forward with choosen action
            state_next, reward, done, info = env.step(action)
            # ^ .unsqueeze(0) turns tensor (*state_space) into a tensor (1, *state_space)

            # Format to pytorch tensors
            state_next = torch.as_tensor(state_next, dtype=torch.float32)
            actions.append(action)
            rewards.append(reward)
            state = state_next
            steps_episode += 1

            if steps_episode > 2000:
                logger.info("Interrupted episode because it exceeded 2,000 steps.")
                break

        end = time.time()

        logger.debug(
            f"--- Computed trajectory in {int(end - start):.2f} s (reward: {sum(rewards)}, steps: {steps_episode})"
        )

        return observations, actions, rewards, steps_episode

    def play_epoch(
        self, env, steps_per_epoch: int, render: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, dict]:
        """
        Samples trajectories inside `env` on-policy until a batch of size at least `steps_per_epoch` is assembled.

        Returns:
            observations: tensor of states observed at t (size (num_steps, **`state_space`))
            actions: tensor of actions taken at t (size (num_steps))
            rewards: tensor of rewards collected at t (size (num_steps))
            weights: tensor of weights by which to multiply the log(policy) terms (size (num_steps));
                     cf. self.compute_weights() for the formulation
            average return: average return over episodes
            info: dict logging run details (e.g. actions, rewards, weigts)
            TODO: complete / structure

        Note the returns have different types than `play_episode`.
        """

        steps_current = 0
        observations = []
        actions = []
        weights = []
        rewards = []
        steps = []
        returns = []  # for total trajectory

        has_rendered_epoch = False

        while steps_current < steps_per_epoch:
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
            ) = self.play_episode(env, render=render_episode, exploit_only=False)
            # ^ exploit_only set to False because play_epoch is only called for learning

            # Compute weights from rewards
            weights_episode = self._compute_weights(obs_episode, rewards_episode)

            # Log run statistics for debugging
            observations.extend(obs_episode)
            actions.extend(actions_episode)
            rewards.extend(rewards_episode)
            steps.extend(list(range(steps_episode)))
            weights.extend(weights_episode)
            returns.append(np.sum(rewards_episode))

            steps_current += steps_episode

        # Store information useful for debugging
        info = {
            "actions": actions,
            "rewards": rewards,
            "weights": weights,
            "steps": steps,
        }

        # Cast observations, actions and weights to tensors for optimisation steps
        # observations: from list of n tensors (*state_space) to tensor batch (n, *state_space)
        # actions: from list of n ints (1) to tensor batch (n)
        # rewards & weights: from list of n floats (1) to tensor batch (n)
        observations = torch.stack(observations)
        actions = torch.as_tensor(actions)
        rewards = torch.tensor(rewards)
        weights = torch.tensor(weights)
        # Compute average return for epoch
        average_return = np.mean(returns)

        return observations, actions, rewards, weights, average_return, info

    def run(
        self,
        env,
        num_epochs: int,
        steps_per_epoch: int,
        save_after_epochs: int,
        print_progress_after: int = 10,
        render: bool = False,
    ) -> pd.DataFrame:
        """
        Updates policy over the course of `num_epoch` epochs of size `steps_per_epoch`,
        saving progress every `save_after_epochs` steps.

        Recommended calls wrap `env` to speed up learning, e.g.:

        ```python
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
        env = make_env(env)  # wrap
        agent = VPGAgent(...)
        agent.run(env, num_epochs=1000, save_after_epochs=1000)
        ```
        """

        if self.save_dir is None:
            logger.warning(
                f"Called method run(), but agent has save dir 'None'. Progress will not be saved."
            )

        infos: List[dict] = []

        logger.info(
            f"Starting training for {num_epochs} epochs with parameters: ... (TODO)"
        )
        start = time.time()

        for epoch in range(num_epochs):
            render_epoch = render and (epoch % print_progress_after == 0)
            # Sample observations and rewards from one epoch
            (
                batch_observations,
                batch_actions,
                batch_rewards,
                batch_weights,
                average_return,
                info,
            ) = self.play_epoch(env, steps_per_epoch, render_epoch)
            # Perform policy upgrade step
            self.optimizer.zero_grad()
            loss_policy = self._compute_loss(
                batch_observations, batch_actions, batch_weights
            )
            loss_policy.backward()
            self.optimizer.step()

            if (epoch > 0) and (epoch % print_progress_after == 0):
                logger.info(
                    f"Epoch: {epoch} \t Returns: {average_return:.2f} \t "
                    f"Steps: {np.mean(info['steps']):.2f} \t Loss: {loss_policy:.2f}"
                )

            if (epoch > 0) and (epoch % save_after_epochs == 0):
                logger.info(f"({epoch}) Saving progress at epoch {epoch}...")
                self._save()
                logger.info("Done.")

            # Log run statistics for debugging
            info["loss_policy"] = loss_policy.item()
            infos.append(info)

        end = time.time()
        logger.info(f"Run complete (runtime: {round(end - start):d} s)")
        logger.info(f"Final average return: {average_return:.2f}")
        if self.save_dir:
            logger.info(f"Saving final state in {str(self.save_dir)}...")
            self._save()
            logger.info("Done.")

        # Format run statistics before returning
        return self._format_info_as_df(infos)

    def _act(self, state: torch.Tensor, exploit_only: bool = False) -> int:
        """
        Samples next action from policy(a|s), either probabilistically if `exploit_only`
        is False (default for learning), or deterministically otherwise.

        Ex.: let a1, a2 be two actions with probability policy(a1|s) = p1
        and policy(a2|s) = p2.
        - `exploit_only` is True: returns a1 with prob. p1 and a2 with prob. p2
        - `exploit_only` is False: returns a1 if p1 > p2, else returns a2
        """
        self.step += 1
        logits = self.policy(state)
        if exploit_only:
            return logits.argmax().item()
        else:
            return Categorical(logits=logits).sample().item()

    def _compute_weights(
        self, states: List[torch.Tensor], rewards: List[np.float64]
    ) -> List[np.float64]:
        """
        Computes weights for every stage transition t in the trajectory as the
        rewards-to-go, i.e.  w(t) = sum_{t'=t}^{T} gamma^{t'-t} * r(t')

        NOTE: `states` is unused in VPG, but used in children classes that
        implement more complex formulation to compute weights.
        """

        # `rewards` contains the reward r(t) collected at every time step
        # We want to compute the sum of (discounted) rewards that each action enabled,
        # not including past rewards; i.e. we want
        #   w(t) = sum_{t'=t}^{T} gamma^{t'-t} * r(t')

        n = len(rewards)
        weights = np.zeros_like(rewards)
        for i in reversed(range(n)):
            weights[i] = rewards[i] + self.gamma * (weights[i + 1] if i + 1 < n else 0)
            # ^ faster than looping forward with weights[i] = rewards[i:].sum()

        return weights

    def _compute_loss(
        self, states: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor
    ) -> torch.float32:
        """
        Computes 'loss' for policy network on a batch of observations.

        Note this is not really a 'loss' in the sense of performance metric, but an objective
        that we train the policy network to maximise (because it has the same gradient
        as the expected return). Minimising the loss does not guarantee improving the
        expected returns, so returns, not losses, should be tracked for monitoring
        performance.

        Params:
            states: list of observations
            actions: list of actions taken (their index)
            weights: weights by which to multiply the log(policy) terms; exact formulation
                     varies, but is returned by self.compute_weights()
        """

        # Check dimensions of states, actions and weights are as expected
        if len(weights.size()) != 1:
            e = f"""
            Expected `weights` to be tensor with single dimension (n), but got tensor with
            size {weights.size()} instead. Passing `weights` with >1 dimension can lead to
            the wrong behaviour when calling `log_prob()`.
            HINT: check dimensions and call .squeeze() if required."
            """
            raise RuntimeError(e)

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

    def _format_info_as_df(self, infos: List[dict]) -> pd.DataFrame:
        """
        Convert run statistics `info` to pd.DataFrame for simpler exploration
        """
        df = []
        for epoch, info in enumerate(infos):
            _df = pd.DataFrame()
            _df["action"] = info["actions"]
            _df["weight"] = info["weights"]
            _df["reward"] = info["rewards"]
            _df["steps"] = info["steps"]
            _df["loss_policy"] = info["loss_policy"]
            _df["epoch"] = epoch
            df.append(_df)
        df = pd.concat(df)
        # Add index for episode
        df["episodes"] = np.where(df["steps"] == 0, 1.0, 0.0).cumsum()  # type: ignore
        return df

    @typing.no_type_check
    def _save(self) -> None:
        """Saves network parameters"""

        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Save current weights
        torch.save(self.policy.state_dict(), self.save_dir / self.save_name)
