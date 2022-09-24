Made with love for my fellow collaborators ❤️

# How to install

## Pre-requisites

To run all functions in this repository, you will need:
- python version >= 3.9
- `poetry` for managing dependencies
- `pre-commit` for running githooks
- `mypy` for type-checking

## Setting up

After installing the dependencies, complete the setup as follows:

1. Clone repository
2. Create your virtual environment: `poetry install`
3. Install pre-commit hooks: `pre-commit install`

You're ready to go! Remember to run every command by appending `poetry run ...` to use the newly created virtual environment. For instance, to start a notebook, use `poetry run jupyter lab`.

# How to run

Instantiate agent classes in module `agent` and environments in `gym` to train your agent and watch them progress through different games.
Here's an example to get started if you just want to watch:

```python
import gym
from agent import VPGGAEAgent

# Create environment
env = gym.make("LunarLander-v2")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Instantiate agent
agent = VPGGAEAgent(
    # Environment params
    state_space=observation_space,
    action_space=action_space,
    # Policy net params
    gamma=0.99,
    lr=0.02,
    policy_net=CategoricalMLP,
    policy_net_kwargs={"input_shape": observation_space, "n_actions": action_space, "hidden_sizes": [32]},
    save_dir=Path("./data/gaecartpole-v1"),
    save_name=Path("policy.pt"),
    # Value func net params
    value_func_net=CategoricalMLP,
    value_func_net_kwargs={"input_shape": observation_space, "n_actions": 1, "hidden_sizes": [32]},
    value_func_lambda=0.96,
    value_func_lr=0.02,
    value_func_save_name=Path("value_func.pt"),
    is_pretrained=False
)

# Train agent for 200 epochs
infos = agent.run(
    env, num_epochs=200, steps_per_epoch=300, save_after_epochs=200, print_progress_after=20, render=False
)

# Watch agent play
(
    obs_episode,
    actions_episode,
    rewards_episode,
    steps_episode,
) = agent.play_episode(env, render=True)
```

# How to contribute

For later :-)
