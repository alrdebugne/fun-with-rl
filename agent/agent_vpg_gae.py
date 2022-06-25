import logging
from pathlib import Path
import time
from typing import Any, List, Tuple, Union

from agent_vpg import VPGAgent

import numpy as np
import pandas as pd
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils import render_in_jupyter

logger = logging.getLogger("vpg-agent")


class VPGGAEAgent(VPGAgent):
    """
    Agent class, using Vanilla Policy Gradient with GAE-lambda advantage estimation
    """

    def __init__(self, gae_lambda: float = 0.98, *super_args, **super_kwargs):
        """ """
        super.__init__(*super_args, **super_kwargs)

        # Add network, loss and optimizer for value function
        self.gae_lambda = gae_lambda
        self.value_func = nn()  # TODO
        self.optimizer_value_func = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr
        )

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
            # Perform policy upgrade step
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch_observations, batch_actions, batch_weights)
            loss.backward()
            self.optimizer.step()

            # ~~~
            # TODO: add update of value function
            # ~~~

            if (epoch > 0) and (epoch % print_progress_after == 0):
                logger.info(
                    f"Epoch: {epoch} \t Returns: {average_return:.2f} \t "
                    f"Steps: {np.mean(info['steps']):.2f} \t Loss: {loss:.2f}"
                )

            if (epoch > 0) and (epoch % save_after_epochs == 0):
                logger.info(f"({epoch}) Saving progress at epoch {epoch}...")
                self._save(dir=self.save_dir, name=self.save_name)  # type: ignore
                logger.info("Done.")

            # Log run statistics for debugging
            info["steps"] = len(batch_observations)
            info["average_return"] = average_return
            info["loss"] = loss.item()
            infos.append(info)

        end = time.time()
        logger.info(f"Run complete (runtime: {round(end - start):d} s)")
        logger.info(f"Final average return: {average_return:.2f}")
        if self.save_dir:
            logger.info(f"Saving final state in {str(self.save_dir)}...")
            self._save(dir=self.save_dir, name=self.save_name)  # type: ignore
            logger.info("Done.")

        # Format run statistics before returning
        return self._format_info_as_df(infos)

    def _compute_weights(
        self, rewards_trajectory: List[np.float64]
    ) -> List[np.float64]:
        """
        Computes weights for every stage transition t in the trajectory using the
        generalised advantage function GAE(gamma, lambda)
        """

        raise NotImplementedError
        # n = len(rewards_trajectory)
        # weights_trajectory = np.zeros_like(rewards_trajectory)
        # for i in reversed(range(n)):
        #     weights_trajectory[i] = rewards_trajectory[i] + self.gamma * (
        #         weights_trajectory[i + 1] if i + 1 < n else 0
        #     )
        #     # ^ faster than looping forward with weights[i] = rewards[i:].sum()

        # return weights_trajectory

    def _compute_loss_value_func(self) -> torch.float32:
        """
        Computes loss for value function a batch of observations.
        """
        raise NotImplementedError
