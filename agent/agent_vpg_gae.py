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
        save_name_value_func: Union[Path, None] = None,
        *super_args,
        **super_kwargs,
    ):
        """ """
        super().__init__(*super_args, **super_kwargs)

        # ~~ Add network, loss and optimizer for value function ~~~
        self._lambda = value_func_lambda
        self.value_func = value_func_net(**value_func_net_kwargs).to(self.device)
        self.optimizer_value_func = torch.optim.Adam(
            self.value_func.parameters(), lr=value_func_lr
        )
        self.save_name_value_func = save_name_value_func

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
            logging.warning(
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
                batch_weights,
                average_return,
                info,
            ) = self.play_epoch(env, steps_per_epoch, render_epoch)

            # Update policy
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch_observations, batch_actions, batch_weights)
            loss.backward()
            self.optimizer.step()

            # Update value function
            self.optimizer_value_func.zero_grad()
            loss_value_func = self._compute_loss_value_func()
            loss_value_func.backward()
            self.optimizer_value_func.step()
            kl_value_func = None  # TODO

            if (epoch > 0) and (epoch % print_progress_after == 0):
                logger.info(
                    f"Epoch: {epoch} \t Returns: {average_return:.2f} \t "
                    f"Steps: {np.mean(info['steps']):.2f} \t Policy loss: {loss:.2f} \t "
                    f"Value funciton loss: {loss_value_func:.2f}"
                )

            if (epoch > 0) and (epoch % save_after_epochs == 0):
                logger.info(f"({epoch}) Saving progress at epoch {epoch}...")
                self._save()
                logger.info("Done.")

            # Log run statistics for debugging
            info["steps"] = len(batch_observations)
            info["average_return"] = average_return
            info["loss"] = loss.item()  # renamed to loss_policy _format_info_as_df
            info["loss_value_func"] = loss_value_func.item()
            info["kl_value_func"] = kl_value_func
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
            deltas[i] = (
                -state_values[i]
                + rewards[i]
                + self.gamma * (state_values[i + 1] if i + 1 < n else 0)
            )

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

    def _compute_loss_value_func(self) -> torch.float32:
        """
        Computes loss for value function a batch of observations.
        """
        raise NotImplementedError

    def _format_info_as_df(self, infos: List[dict]) -> pd.DataFrame:
        """
        Based on VPGAgent._format_info_as_df(). Renames `loss` to `loss_policy`
        and adds statistics for value function updates.
        """
        df = super()._format_info_as_df(infos)
        df.rename(columns={"loss": "loss_policy"}, inplace=True)

        # Add statistics for value function updates
        d_loss, d_kl = {}, {}
        for epoch, info in enumerate(infos):
            d_loss[epoch] = info["loss_value_func"]
            d_kl[epoch] = info["kl_value_func"]

        df["loss_value_func"] = df["epoch"].map(d_loss)
        df["kl_value_func"] = df["epoch"].map(d_kl)

        return df

    @typing.no_type_check
    def _save(self) -> None:
        """Saves network parameters"""

        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Save current weights
        torch.save(self.policy.state_dict(), self.save_dir / self.save_name)
        torch.save(
            self.value_func.state_dict(), self.save_dir / self.save_name_value_func
        )
