"""
Run file
TODO: delete because deprecated
"""

import click
import logging
import pickle
from tqdm import tqdm
from typing import Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from agent import DDQNAgent
from wrappers import wrappers


def make_env(env, actions: Optional[str] = None):
    """
    Simplify screen following original Atari paper
    TODO: move to wrappers.py (or somewhere else with wrappers.py)
    """
    env = wrappers.MaxAndSkipEnv(env)  # repeat action over four frames
    env = wrappers.ProcessFrame84(env)  # size to 84 * 84 and greyscale
    env = wrappers.ImageToPyTorch(env)  # convert to (C, H, W) for pytorch
    env = wrappers.BufferWrapper(env, 4)  # stack four frames in one 'input'
    env = wrappers.ScaledFloatFrame(env)  # normalise RGB values to [0, 1]
    actions = actions or RIGHT_ONLY
    if not actions in [RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT]:
        e = (
            "`actions` must be one of RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, "
            f"but received {actions} instead."
        )
        raise ValueError(e)
    return JoypadSpace(env, actions)


@click.command()
@click.option("--lr", default=0.002, help="Learning rate")
@click.option("--num_episodes", default=10000, help="Number of learning episodes")
@click.option("--save_dir", default="./", help="Path for saving model iterations")
@click.option(
    "--pretrained",
    default=False,
    help="Whether model should resume learning from past save",
)
def train(lr: float, num_episodes: int, save_dir: str, pretrained: bool):
    """
    Main run function to train agent
    """

    # Create simplified environment for Super Mario Bros
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env = make_env(env)

    # Initialize agent
    agent = DDQNAgent(
        state_space=env.observation_space.shape,
        action_space=env.action_space.n,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.9,
        lr=lr,
        dropout=0.0,
        exploration_max=1.0,
        exploration_min=0.2,
        exploration_decay=0.99,
    )

    # Code run step
    is_training = True
    env.reset()
    final_rewards = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor(np.array([state]))
        reward_episode = 0
        steps_episode = 0
        done = False

        while not done:

            action = agent.act(state)
            steps_episode += 1
            state_next, reward, done, info = env.step(int(action[0]))
            reward_episode += reward

            # Format to pytorch tensors
            state_next = torch.Tensor(np.array([state_next]))
            reward = torch.tensor([reward]).unsqueeze(0)
            done = torch.tensor([int(done)]).unsqueeze(0)

            if is_training:
                agent.remember(state, action, reward, state_next, done)
                agent.experience_replay()

            state = state_next

        print(f"Final reward after episode {episode}: {reward_episode:.2f}")

        # Record reward achieved in n-th episode
        final_rewards.append(reward_episode)

    env.close()


if __name__ == "__main__":
    train()
