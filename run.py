"""
Run file
TODO: add description
"""

import click
import logging
import pickle
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from agent import DDQNAgent
from wrappers import wrappers


def make_env(env):
    """Simplify screen following original Atari paper"""
    env = wrappers.MaxAndSkipEnv(env)  # repeat action over four frames
    env = wrappers.ProcessFrame84(env)  # size to 84 * 84 and greyscale
    env = wrappers.ImageToPyTorch(env)  # convert to (C, H, W) for pytorch
    env = wrappers.BufferWrapper(env, 4)  # stack four frames in one 'input'
    env = wrappers.ScaledFloatFrame(env)  # normalise RGB values to [0, 1]
    return JoypadSpace(env, RIGHT_ONLY)


def save_progress(dir: Path, agent: DDQNAgent, rewards: list):
    """Saves network states and parameters to resume training"""
    # TODO: move to method from agent?

    with open(dir / Path("memory_pointer.pkl"), "wb") as f:
        pickle.dump(agent.memory_pointer, f)
    with open(dir / Path("memory_num_experiences.pkl"), "wb") as f:
        pickle.dump(agent.memory_num_experiences, f)
    try:
        with open(dir / Path("rewards.pkl"), "wb") as f:
            pickle.dump(rewards, f)
    except:
        pass
        # TODO: fix: rewards.pkl doesn't get created inside agent
    with open(dir / Path("lr.pkl"), "wb") as f:
        pickle.dump(agent.lr, f)
    with open(dir / Path("exploration_rate.pkl"), "wb") as f:
        pickle.dump(agent.exploration_rate, f)

    torch.save(agent.primary_net.state_dict(), dir / Path("dq_primary.pt"))
    torch.save(agent.target_net.state_dict(), dir / Path("dq_target.pt"))
    torch.save(agent.STATE_MEM, dir / Path("STATE_MEM.pt"))
    torch.save(agent.ACTION_MEM, dir / Path("ACTION_MEM.pt"))
    torch.save(agent.REWARD_MEM, dir / Path("REWARD_MEM.pt"))
    torch.save(agent.STATE2_MEM, dir / Path("STATE2_MEM.pt"))
    torch.save(agent.DONE_MEM, dir / Path("DONE_MEM.pt"))


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
    TODO: docstring
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
        state = torch.Tensor([state])
        reward_episode = 0
        steps_episode = 0
        done = False

        while not done:

            action = agent.act(state)
            steps_episode += 1
            state_next, reward, done, info = env.step(int(action[0]))
            reward_episode += reward

            # Format to pytorch tensors
            state_next = torch.Tensor([state_next])
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
