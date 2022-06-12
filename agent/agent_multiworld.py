import logging
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

from .agent import DDQNAgent

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ddqn-multiworld-agent")


class MultiworldDDQNAgent(DDQNAgent):
    """
    Builds on top of DDQNAgent to allow learning across multiple environments at once.
    The memory buffer is apportioned in N chunks, each dedicated to one env at a time,
    to avoid catastrophic forgetting (with N is the number of environments).

    TODO: make sampling of indices in `recall` robust.
    Currently, recall samples from range(self.memory_num_experiences). This can point to
    empty indices if we switch to another env before its apportioned chunk of memory is
    filled up.
    """

    def __init__(
        self,
        n_envs: int,
        *super_args,
        **super_kwargs,
    ):
        super(MultiworldDDQNAgent, self).__init__(*super_args, **super_kwargs)
        self.n_envs = n_envs
        # Create helper memory pointers to apportion the memory in equal chunks for every env
        self.memory_pointer_per_env = {i: 0 for i in range(n_envs)}
        self.max_memory_size_per_env = self.max_memory_size / n_envs
        if not self.max_memory_size_per_env.is_integer():
            self.max_memory_size_per_env = int(math.floor(self.max_memory_size_per_env))
            logging.warning(
                f"Memory size {self.max_memory_size} cannot be divided in {n_envs} equal chunks. "
                f"Setting memory size to {self.max_memory_size_per_env} per env instead."
            )

    # Getter
    def get_env_index(self) -> int:
        return self._env_index

    # Setter
    def set_env_index(self, val: int) -> None:
        self._env_index = int(val)

    # Property
    env_index = property(get_env_index, set_env_index)

    def update_memory_pointer_and_count(self) -> None:
        """
        Updates index for memory pointer and number of experiences in memory, apportioning
        a contiguous buffer of size `max_memory_size_per_env` for each env.
        Must be called _before_ each call to `remember`.

        TODO: set env_index inside class instead of outside (requires refactoring)
        """
        # Update position of pointer inside chunk dedicated to env
        env_index = self.env_index
        # FIXME: ^ set outside agent in run loop...
        self.memory_pointer_per_env[env_index] = int(
            (self.memory_pointer_per_env[env_index] + 1) % self.max_memory_size_per_env
        )
        # Update global pointer, taking into account starting position
        self.memory_pointer = int(
            env_index * self.max_memory_size_per_env
            + self.memory_pointer_per_env[env_index]
        )
        self.memory_num_experiences = min(
            self.memory_num_experiences + 1, self.max_memory_size
        )

    def run(  # type: ignore[override]
        self,
        envs,
        num_episodes: int,
        save_step: int,
        cycle_env_after: int,
        print_progress_after: int = 50,
    ) -> Tuple[Dict[int, List[float]], Dict[int, List[int]]]:
        """
        Multiworld version of DDQNAgent.run()

        Again, it is recommended calls wrap `env` to speed up learning, e.g.:

        ```python
        envs = [
            make_env(
                gym_super_mario_bros.make(f"SuperMarioBros-{world}-{level}-v3")
            ) for world in range(1, 3) for level in range(1, 5)
        ]
        agent = MultiworldDDQNAgent(...)
        agent.run(envs, num_episodes=10000, save_step=1000)
        ```
        """

        if not isinstance(self.save_dir, Path):
            e = f"Called method run(), but agent has invalid save dir: {self.save_dir}"
            raise ValueError(e)

        logger.info(
            f"Starting training for {num_episodes} episodes across {self.n_envs} environments with parameters: ... (TODO)"
        )
        start = time.time()

        rewards_all_per_env: dict = {i: [] for i in range(len(envs))}
        steps_all_per_env: dict = {i: [] for i in range(len(envs))}

        for episode in range(num_episodes):
            env_index = (episode // cycle_env_after) % self.n_envs
            # ^ cycling through to the next environment after `cycle_env_after` episodes
            env = envs[env_index]  # .reset() called in .play_episode()
            self.env_index = env_index

            reward_episode, steps_episode = self.play_episode(env, is_training=True)
            # ^ no point in calling .run() if not training
            rewards_all_per_env[env_index].append(reward_episode)
            steps_all_per_env[env_index].append(steps_episode)

            if (episode > 0) & (episode % print_progress_after == 0):
                logger.info(
                    f"[Env {env_index}] Reward after episode {episode}: {reward_episode:.2f} ({steps_episode} steps)"
                )

            if (episode > 0) & (episode % save_step == 0):
                logger.info(f"Saving progress at episode {episode}...")
                self.save(dir=self.save_dir)
                logger.info(f"Done.")

        end = time.time()
        logger.info(f"Run complete (runtime: {round(end - start):d} s)")
        logger.info(f"Final reward: {reward_episode}")
        logger.info("Saving final state...")
        self.save(dir=self.save_dir)
        logger.info("Done.")

        return rewards_all_per_env, steps_all_per_env
