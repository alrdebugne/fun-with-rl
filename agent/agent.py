import logging
from pathlib import Path
import pickle
import random
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .solver import DQNetwork

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ddqn-agent")


class DDQNAgent:
    """
    Agent class, using double Q-learning (https://arxiv.org/pdf/1509.06461.pdf)
    Main methods:
    - `remember`: save trajectories to buffer, to batch learn from them
    - `recall`: sample a batches of trajectories at random
    - `act`: pick agent's next action, either by using its policy network, or through exploration
    - `experience_replay`: sample transitions from memory buffer & update weights
    """

    def __init__(
        self,
        state_space: npt.NDArray[np.float64],
        action_space: int,
        max_memory_size: int,
        batch_size: int,
        gamma: float,
        lr: float,
        dropout: float,
        exploration_max: float,
        exploration_min: float,
        exploration_decay: float,
        save_dir: Path = Path("./data/tmp/"),
        is_pretrained: bool = False,
    ):
        """ """
        # ~~~ Define layers for DDQN ~~~
        self.state_space = state_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        self.is_pretrained = is_pretrained
        # Since this is Double Q-Learning, we instantiate two networks: one local for policies, one target for XX
        self.primary_net = DQNetwork(state_space, action_space, dropout).to(self.device)
        self.target_net = DQNetwork(state_space, action_space, dropout).to(self.device)
        if self.is_pretrained:
            if self.save_dir is None:
                raise ValueError("`save_dir` must be specified for resuming training")
            logger.info("Loading weights from previous runs...")
            # Load weights from previous iteration
            self.primary_net.load_state_dict(
                torch.load(
                    self.save_dir / Path("dq_primary.pt"),
                    map_location=torch.device(self.device),
                )
            )
            self.target_net.load_state_dict(
                torch.load(
                    self.save_dir / Path("dq_target.pt"),
                    map_location=torch.device(self.device),
                )
            )
        self.optimizer = torch.optim.Adam(self.primary_net.parameters(), lr=lr)
        # ^ keep default params for now: lr = 0.001, betas = (0.9, 0.999), eps = 1e-8
        self.copy = 5000  # copy local weights to target network after `copy` steps
        self.step = 0  # initialise steps

        # ~~~ Create memory ~~~
        # (for experience replay)
        self.max_memory_size = max_memory_size
        self.memory_sample_size = batch_size

        if self.is_pretrained:
            logger.info("Loading memory from previous runs...")
            self.STATE_MEM = torch.load(self.save_dir / Path("STATE_MEM.pt"))
            self.STATE2_MEM = torch.load(self.save_dir / Path("STATE2_MEM.pt"))
            self.ACTION_MEM = torch.load(self.save_dir / Path("ACTION_MEM.pt"))
            self.REWARD_MEM = torch.load(self.save_dir / Path("REWARD_MEM.pt"))
            self.DONE_MEM = torch.load(self.save_dir / Path("DONE_MEM.pt"))
            with open(self.save_dir / Path("memory_pointer.pkl"), "rb") as f:
                self.memory_pointer = pickle.load(f)
            with open(self.save_dir / Path("memory_num_experiences.pkl"), "rb") as f:
                self.memory_num_experiences = pickle.load(f)
            with open(self.save_dir / Path("exploration_rate.pkl"), "rb") as f:
                self.exploration_rate = pickle.load(f)
            with open(self.save_dir / Path("rewards.pkl"), "rb") as f:
                self.rewards = pickle.load(f)
            with open(self.save_dir / Path("steps.pkl"), "rb") as f:
                self.steps = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.memory_pointer = 0  # pointer in memory buffer
            self.memory_num_experiences = 0  # number of experiences in memory
            self.exploration_rate = exploration_max  # initialise
            self.rewards = []  # initialise
            self.steps = []  # initialise

        # ~~~ Learning parameters ~~~
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss(beta=1.0).to(self.device)
        # ^ Huber loss, i.e. quadratic if |y - y_hat| < beta, else L1
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def update_memory_pointer_and_count(self) -> None:
        """
        Updates index for memory pointer and number of experiences in memory.
        Must be called _before_ each call to `remember`.
        """
        self.memory_pointer = (self.memory_pointer + 1) % self.max_memory_size
        self.memory_num_experiences = min(
            self.memory_num_experiences + 1, self.max_memory_size
        )

    def remember(self, state, action, reward, state2, done) -> None:
        """
        Store experience tuples (S, A, R, S') in memory for experience replay.
        Must call `update_memory_pointer_and_count` _before_ each call to `remember`.
        """
        self.STATE_MEM[self.memory_pointer] = state.float()
        self.ACTION_MEM[self.memory_pointer] = action.float()
        self.REWARD_MEM[self.memory_pointer] = reward.float()
        self.STATE2_MEM[self.memory_pointer] = state2.float()
        self.DONE_MEM[self.memory_pointer] = done.float()

    def recall(self) -> Tuple:
        """Sample experience tuple (S, A, R, S') from memory buffer"""
        # Randomly sample a batch of `memory_sample_size` experiences
        indices = random.choices(
            range(self.memory_num_experiences), k=self.memory_sample_size
        )
        states = self.STATE_MEM[indices]
        actions = self.ACTION_MEM[indices]
        rewards = self.REWARD_MEM[indices]
        states2 = self.STATE2_MEM[indices]
        dones = self.DONE_MEM[indices]

        return states, actions, rewards, states2, dones

    def act(self, state) -> torch.Tensor:
        """
        Epsilon-greedy policy: output next action given `state`, either by:
        - exploring randomly, with proba. `self.exploration_rate`
        - picking best action from primary net
        """
        # Format s.t. the action is returned as a 'nested' torch.Tensor, e.g. tensor([[3]])
        # TODO: why that format?
        self.step += 1
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        return (
            torch.argmax(self.primary_net(state.to(self.device)))
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
            # ^ calling unsqueeze to return 'nested' tensor
        )

    def copy_target_to_primary(self) -> None:
        """Copy weights from target net to primary net"""
        self.target_net.load_state_dict(self.primary_net.state_dict())

    def experience_replay(self):
        """
        Sample experiences from memory buffer to update weights.

        Experience replay is preferred over step-by-step learning (mostly) to break correlations
        between adjacent frames.
        """
        # Update target model weights every `self.copy` steps
        if (self.step % self.copy == 0) and not (self.is_pretrained and self.step == 0):
            self.copy_target_to_primary()

        # Only start experience replay once there are more experiences than batch size
        if self.memory_sample_size > self.memory_num_experiences:
            return

        # Recall batches of experience, selected at random
        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        # ^ set gradients to zero (by default, pytorch accumulates gradients unless reset)
        # see https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        # For sample of experiences, compute:
        # - observed rewards, i.e. the target
        # - predicted rewards, i.e. the model output
        target = REWARD + torch.mul(
            self.gamma * self.target_net(STATE2).max(1).values.unsqueeze(1),
            1 - DONE,
        )
        # ^ multiplication with (1 - DONE) vectorises the condition that, if DONE,
        # then target = reward

        # Note on syntax (for torch newbies):
        # - `gamma * net(state)` returns the discounted predicted rewards over the action space
        #   (e.g. tensor([[-0.0337,  0.0123,  0.0229, -0.0087, -0.0145]]))
        # - calling `.max(1).values` returns the maximum over dimension 1, i.e. our max. reward
        #   (e.g. tensor([0.0229])). `.values` is required because `.max(dim)` returns
        #   a namedtuple (values, indices).
        # - .unsqueeze(1) formats to 'nested' tensor again (e.g. tensor([[0.0229]])
        # So the code above is equivalent to calling `gamma * net(state).max().unsqueeze(0)`,
        # but... I suppose it's good practice to specify the dimension of the `max`.

        pred = self.primary_net(STATE).gather(1, ACTION.long())
        # ^ fetch ('gather') index ACTION within the outcome distribution
        # .long() converts ACTION to int64 -- not sure why that's needed

        # Compute loss, gradient & update parameters through back-propagation
        loss = self.l1(pred, target)
        loss.backward()  # compute gradients
        self.optimizer.step()  # update parameters

        # Adjust exploration rate
        self.exploration_rate = max(
            self.exploration_rate * self.exploration_decay, self.exploration_min
        )

    def play_episode(self, env, is_training: bool) -> Tuple[float, int]:
        """
        Plays one episode from start to finish in `env` and returns the associated reward.
        If `is_training` is True, also updates weights through experience replay.

        NOTE: `env` must be compatible with the agent's `observation_space` and `action_space`
        attributes. `env` is left as an argument to allow the agent to learn on several levels at once.
        """

        state = env.reset()
        state = torch.Tensor(np.array([state]))
        reward_episode = 0
        steps_episode = 0
        done = False

        while not done:
            action = self.act(state)
            steps_episode += 1
            state_next, reward, done, info = env.step(int(action[0]))
            reward_episode += reward

            # Format to pytorch tensors
            state_next = torch.Tensor(np.array([state_next]))
            reward = torch.tensor([reward]).unsqueeze(0)
            done = torch.tensor([done]).unsqueeze(0)

            if is_training:
                self.update_memory_pointer_and_count()
                self.remember(state, action, reward, state_next, done)
                self.experience_replay()
                # TODO: update methods `update_...` and `remember` s.t. index 0 isn't skipped
                # at first iteration, while remaining compatible with MultiworldDDQNAgent.

            state = state_next

        return reward_episode, steps_episode

    def save(self, dir: Path):
        """Saves memory buffer and network parameters"""

        dir.mkdir(parents=True, exist_ok=True)

        with open(dir / Path("memory_pointer.pkl"), "wb") as f:
            pickle.dump(self.memory_pointer, f)
        with open(dir / Path("memory_num_experiences.pkl"), "wb") as f:
            pickle.dump(self.memory_num_experiences, f)
        with open(dir / Path("exploration_rate.pkl"), "wb") as f:
            pickle.dump(self.exploration_rate, f)
        with open(dir / Path("rewards.pkl"), "wb") as f:
            pickle.dump(self.rewards, f)
        with open(dir / Path("steps.pkl"), "wb") as f:
            pickle.dump(self.steps, f)

        torch.save(self.primary_net.state_dict(), dir / Path("dq_primary.pt"))
        torch.save(self.target_net.state_dict(), dir / Path("dq_target.pt"))
        torch.save(self.STATE_MEM, dir / Path("STATE_MEM.pt"))
        torch.save(self.ACTION_MEM, dir / Path("ACTION_MEM.pt"))
        torch.save(self.REWARD_MEM, dir / Path("REWARD_MEM.pt"))
        torch.save(self.STATE2_MEM, dir / Path("STATE2_MEM.pt"))
        torch.save(self.DONE_MEM, dir / Path("DONE_MEM.pt"))

    def run(
        self,
        env,
        num_episodes: int,
        save_step: int,
        print_progress_after: int = 50,
    ) -> None:
        """
        Plays `num_episodes` games, saving progress every `save_step` steps.

        Recommended calls wrap `env` to speed up learning, e.g.:

        ```python
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
        env = make_env(env)  # wrap
        agent = DDQNAgent(...)
        agent.run(env, num_episodes=10000, save_step=1000)
        ```
        """

        if not isinstance(self.save_dir, Path):
            e = f"Called method run(), but agent has invalid save dir: {self.save_dir}"
            raise ValueError(e)

        logger.info(
            f"Starting training for {num_episodes} episodes with parameters: ... (TODO)"
        )
        start = time.time()

        for episode in range(num_episodes):
            reward_episode, steps_episode = self.play_episode(env, is_training=True)
            # ^ no point in calling .run() if not training
            self.rewards.append(reward_episode)
            self.steps.append(steps_episode)

            if (episode > 0) & (episode % print_progress_after == 0):
                logger.info(
                    f"Reward after episode {episode}: {reward_episode:.2f} ({steps_episode} steps)"
                )

            if (episode > 0) & (episode % save_step == 0):
                logger.info(f"Saving progress at episode {episode}...")
                self.save(dir=self.save_dir)
                logger.info("Done.")

        end = time.time()
        logger.info(f"Run complete (runtime: {round(end - start):d} s)")
        logger.info(f"Final reward: {reward_episode}")
        logger.info("Saving final state...")
        self.save(dir=self.save_dir)
        logger.info("Done.")
