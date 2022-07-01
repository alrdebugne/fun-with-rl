import logging
from pathlib import Path
import time
import typing
from typing import Any, List, Tuple, Union

from agent import VPGAgent

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger("vpg-agent")


class VPGGAEAgent(VPGAgent):
    """
    Agent class, using Vanilla Policy Gradient with GAE-lambda advantage estimation
    """

    def __init__(
        self,
        value_func_net: nn.Module,
        value_func_net_kwargs: dict,
        value_func_lambda: float,
        value_func_lr: float,
        value_func_save_name: Union[Path, None] = None,
        *super_args,
        **super_kwargs,
    ):
        """ """
        super().__init__(*super_args, **super_kwargs)

        # ~~ Add network, loss and optimizer for value function ~~~
        self._lambda = value_func_lambda
        self.value_func = value_func_net(**value_func_net_kwargs).to(self.device)
        self.value_func_loss = nn.MSELoss(reduction="mean")
        self.value_func_optimizer = torch.optim.Adam(
            self.value_func.parameters(), lr=value_func_lr
        )
        self.value_func_iter_per_epoch = 50
        # ^ how many times we update the value function on the same batch of observations
        # Heuristically, updating multiple times with lower learning rate seems to lead
        # to stabler learning
        self.value_func_save_name = value_func_save_name
        self.kl_div = nn.KLDivLoss(reduce="batchmean")  # for monitoring learning

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
            _losses_value_func = []  # for logging; reset every epoch

            # Sample observations and rewards from one epoch
            (
                batch_observations,
                batch_actions,
                batch_rewards,
                batch_weights,
                average_return,
                info,
            ) = self.play_epoch(env, steps_per_epoch, render_epoch)

            # Update policy
            self.optimizer.zero_grad()
            loss_policy = self._compute_loss(
                batch_observations, batch_actions, batch_weights
            )
            loss_policy.backward()
            self.optimizer.step()

            # Update value function
            for _ in range(self.value_func_iter_per_epoch):
                self.value_func_optimizer.zero_grad()
                loss_value_func = self._compute_loss_value_func(
                    batch_observations, batch_rewards
                )
                loss_value_func.backward()
                self.value_func_optimizer.step()
                # Log learning:
                _losses_value_func.append(loss_value_func.item())
                # ^ Not the cleverest way of storing, since this duplicates losses for all steps
                # of the trajectory...
                # kl_value_func = self.kl_div()

            if (epoch > 0) and (epoch % print_progress_after == 0):
                logger.info(
                    f"Epoch: {epoch} \t Returns: {average_return:.2f} \t "
                    f"Steps: {np.mean(info['steps']):.2f} \t Policy loss: {loss_policy:.2f} \t "
                    f"Value function loss: {loss_value_func.item():.2f}"
                )

            if (epoch > 0) and (epoch % save_after_epochs == 0):
                logger.info(f"({epoch}) Saving progress at epoch {epoch}...")
                self._save()
                logger.info("Done.")

            # Log run statistics for debugging
            info["loss_policy"] = loss_policy.item()
            info["loss_value_func"] = _losses_value_func
            # info["kl_value_func"] = kl_value_func
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

    def _bellman_residuals(self, states: List[torch.Tensor], rewards: List[np.float64]):
        """
        Returns array Bellman residuals (d(t1), d(t2), ..., d(tn-1), where
            d(t) = -V(s_t) + r(t) + gamma * V(s_{t+1}) is the t-th Bellman residual,
            V(s_t) is the value function evaluated at t,
            r(t) is the reward at step t,
            gamma is a discount factor.
        """
        # Compute value functions V(s_t) for all states
        state_values = self.value_func(torch.stack(states))
        # ^ .stack converts list of n tensors (*state_space) to tensor (n, *state_space)

        # Compute Bellman residuals
        n = len(rewards)
        deltas = np.zeros_like(rewards)
        for i in range(n):
            # fmt: off
            deltas[i] = -state_values[i] + rewards[i] + self.gamma * (
                state_values[i + 1] if i + 1 < n else 0
            )
            # fmt: on

        return deltas

    def _compute_weights(
        self, states: List[torch.Tensor], rewards: List[np.float64]
    ) -> List[np.float64]:
        """
        Returns weights using generalised advantage function GAE(gamma, lambda) as:
            weight[t] = sum_{t'=t}^{T} (gamma*lambda)^{t'-t} * d(t'),
        where d(t') is the Bellman residual discounted at rate gamma.
        """

        deltas = self._bellman_residuals(states, rewards)
        #
        weights = np.zeros_like(rewards)
        n = len(rewards)
        for i in reversed(range(n)):
            weights[i] = deltas[i] + self.gamma * self._lambda * (
                weights[i + 1] if i + 1 < n else 0
            )

        return weights

    def _compute_loss_value_func(
        self, states: List[torch.Tensor], rewards: List[np.float64]
    ) -> torch.float32:
        """
        Computes the L2 loss of our value function over a batch of observations as
            loss = sum( [value_func(s_t) - value_observed(s_t)] ** 2 ),
        where
            value_func are the predictions made by our model,
            value_observed are the target rewards, calculated as the discounted
            sum of rewards-to-go
        """
        # Compute predictions of value model V(s_t) for all states
        predictions = self.value_func(states).squeeze()
        # Define target values as the discounted sum of rewards-to-go
        targets = torch.as_tensor(
            super()._compute_weights(states, rewards), dtype=torch.float32
        )
        # ^ super's weights are discounted rewards-to-go (cast to torch.tensor)

        # Return L2 loss (summed)
        return self.value_func_loss(predictions, targets)

    def _format_info_as_df(self, infos: List[dict]) -> pd.DataFrame:
        """
        Based on VPGAgent._format_info_as_df(). Adds statistics for value function updates.
        """
        df = super()._format_info_as_df(infos)

        # Add statistics for value function updates
        d_loss, d_kl = {}, {}
        for epoch, info in enumerate(infos):
            d_loss[epoch] = info["loss_value_func"]
            d_kl[epoch] = info.get("kl_value_func", None)  # temporary

        df["loss_value_func"] = df["epoch"].map(d_loss)
        df["kl_value_func"] = df["epoch"].map(d_kl)

        return df

    @typing.no_type_check
    def _save(self) -> None:
        """Saves network parameters"""

        self.save_dir.mkdir(parents=True, exist_ok=True)
        # TODO: add log for hyperparameters
        # Save current weights
        torch.save(self.policy.state_dict(), self.save_dir / self.save_name)
        torch.save(
            self.value_func.state_dict(), self.save_dir / self.value_func_save_name
        )
