# -*- coding: utf-8 -*-
from copy import deepcopy
import pytest

from agent import MultiworldDDQNAgent, CategoricalCNN


agent_default_kwargs = {
    "state_space": (4, 84, 84),
    "action_space": 7,
    "batch_size": 32,
    "gamma": 0.9,
    "lr": 0.001,
    "policy_net": CategoricalCNN,
    "policy_net_kwargs": {},
    "exploration_max": 1.0,
    "exploration_min": 0.05,
    "exploration_decay": 0.999,
    "save_dir": "",
    "is_pretrained": False,
}


@pytest.mark.parametrize(
    "n_envs, max_memory_size, num_episodes, num_episodes_per_cycle, num_steps_per_episode, exp",
    # fmt: off
    [
        (4, 20, 4, 1, 5, [1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 16, 17, 18, 19, 15]),
        # ^ 4 environments, one playthrough each, just fits in memory
        # Note how the first index for each cycle is skipped: commented in DDQNAgent.play_episode
        (4, 20, 4, 1, 7, [1, 2, 3, 4, 0, 1, 2, 6, 7, 8, 9, 5, 6, 7, 11, 12, 13, 14, 10, 11, 12, 16, 17, 18, 19, 15, 16, 17]),
        (8, 24, 8, 1, 1, [1, 4, 7, 10, 13, 16, 19, 22]),
        (8, 24, 8, 1, 2, [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23]),
        (16, 32000, 4, 1, 3, [1, 2, 3, 2001, 2002, 2003, 4001, 4002, 4003, 6001, 6002, 6003],
        ),
    ],
    # fmt: on
)
def test_update_memory_pointer_and_count(
    n_envs,
    max_memory_size,
    num_episodes,
    num_episodes_per_cycle,
    num_steps_per_episode,
    exp,
) -> None:
    """
    Test MultiworldDDQNAgent.update_memory_pointer_and_count by faking playing through episodes,
    then checking how the memory pointer is updated.
    """
    # Instantiate agent with current test's args
    _kwargs = deepcopy(agent_default_kwargs)
    _kwargs["n_envs"] = n_envs
    _kwargs["max_memory_size"] = max_memory_size
    _agent = MultiworldDDQNAgent(**_kwargs)  # type: ignore

    # Fake playing through episode, storing the memory pointer at each step
    res = []
    for episode in range(num_episodes):
        steps = 0  # reset steps
        env_index = (episode // num_episodes_per_cycle) % _agent.n_envs
        _agent.env_index = env_index
        for _ in range(num_steps_per_episode):
            steps += 1
            _agent._update_memory_pointer_and_count()
            res.append(_agent.memory_pointer)

    assert res == exp, f"Expected {exp}, but received {res}"
