# -*- coding: utf-8 -*-
"""
Module that tests that VPGAgent performs well on simple tasks,
e.g. Cartpole.

"""
from copy import deepcopy
import pytest
import gym

from agent import SimpleMLPNetwork, VPGAgent

# # env = gym.make("CartPole-v1")
# env = gym.make("LunarLander-v2")
# observation_space = env.observation_space.shape[0]
# action_space = env.action_space.n

vpgagent_kwargs = {
    "gamma": 1.0,
    "lr": 0.02,
    "policy_net": SimpleMLPNetwork,
    "policy_net_kwargs": {"hidden_sizes": [32]},
}
vpg_agent_run_kwargs_cartpole = {
    "num_epochs": 100,
    "steps_per_epoch": 1000,
    "save_after_epochs": 9999,
    "print_progress_after": 10,
    "render": False,
}


@pytest.mark.parametrize(
    "env_name, agent_class, agent_kwargs, agent_run_kwargs, min_avg_score",
    # fmt: off
    [
        ("CartPole-v1", VPGAgent, vpgagent_kwargs, vpg_agent_run_kwargs_cartpole, 450),
    ],
    # fmt: on
)
def test_vpg_agent_on_simple_envs(
    env_name, agent_class, agent_kwargs, agent_run_kwargs, min_avg_score
) -> None:
    """
    Test that different VPG implementations on simple environments
    """
    # Start environment
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Instantiate agent
    _kwargs = deepcopy(agent_kwargs)
    _kwargs["state_space"] = observation_space
    _kwargs["action_space"] = action_space
    agent = agent_class(**_kwargs)  # type: ignore

    # Learn to play
    infos = agent.run(env=env, **agent_run_kwargs)
    # Get average score over the last 10 epochs
    last_tenth_epoch = max(0, agent_run_kwargs["num_epochs"] - 10)
    avg_score = (
        infos.loc[infos["epoch"] >= last_tenth_epoch]
        .groupby("epoch")["average_return"]
        .mean()
        .mean()
    )

    assert (
        avg_score > min_avg_score
    ), f"Expected min. score of  {min_avg_score}, but received {avg_score} over epochs {last_tenth_epoch} to {last_tenth_epoch+10}"
