# TODO: refactor into utils (general) and agent/utils (RL-specific)

import collections
from IPython import display
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
from pathlib import Path
import random
import torch
from typing import *


def set_seed(seed: int, env):
  """ Sets seed everywhere I can think of for reproducibility """
  logging.info(f"Setting seed {seed}")
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  env.seed(seed)
  env.action_space.seed(seed)


def get_n_trainable_params(model: torch.nn.Module) -> int:
    # Get all trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Sum their dimensions
    n_trainable_params = sum(
        np.prod(p.cpu().detach().numpy().shape) for p in trainable_params
        # Needs to be on CPU and detached ¯\_(ツ)_/¯
    )
    return n_trainable_params


def decay_lr_linearly(step: int, decay_steps: int, lr_max: float, lr_min: float):
    """
    Decays linearly until lr_min over decay_steps
    """
    return max(lr_min, lr_max - (lr_max - lr_min) / decay_steps * step)


def get_fancy_decay_factor(step: int, total_steps: int) -> float:
    """
    Returns a multiplicative factor that anneals towards 0, used for
    learning or exploration rate decay.

    Based on https://discovery.ucl.ac.uk/id/eprint/10056194/1/Diagnosis%20and%20referral%20in%20retinal%20disease%20-%20updated.pdf
    """

    if step < 0.1 * total_steps:
        return 1.
    elif step < 0.2 * total_steps:
        return 1 / 2
    elif step < 0.5 * total_steps:
        return 1 / 4
    elif step < 0.7 * total_steps:
        return 1 / 8
    elif step < 0.9 * total_steps:
        return 1 / 64
    elif step < 0.95 * total_steps:
        return 1 / 256
    else:
        return 1 / 512


def moving_average(a, n):
    """ """
    rolling = pd.Series(a).rolling(window=n)
    return rolling.mean(), rolling.std()


def plot_learning_trajectories(returns: list, losses: list, qvals: list, subplots_kwargs: dict):
    """ """
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, **subplots_kwargs)

    # Returns
    x = range(len(returns))
    returns_avg, returns_std = moving_average(np.array(returns), 20)
    ax1.plot(x, returns_avg, color="b")
    ax1.fill_between(x, (returns_avg - returns_std), (returns_avg + returns_std), color="b", alpha=.1)
    ax1.set_title("Returns (every episode)");

    # Losses
    x = range(len(losses))
    losses_avg, losses_std = moving_average(np.array(losses), 200)
    ax2.plot(x, losses_avg, color="r")
    ax2.fill_between(x, (losses_avg - losses_std), (losses_avg + losses_std), color="r", alpha=.1)
    ax2.set_title("Losses (every learning iteration)");

    # Q-values (either average or max, based on what user enters)
    x = range(len(qvals))
    qvals_avg, qvals_std = moving_average(np.array(qvals), 20)
    ax3.plot(x, qvals_avg, color="g")
    ax3.fill_between(x, (qvals_avg - qvals_std), (qvals_avg + qvals_std), color="g", alpha=.1)
    ax3.set_title("Average Q-values (on hold-out)");
    return f


def save_animation(env, agent, device: str, fpath: Union[str, Path]) -> None:
    """ """

    fig = plt.figure()
    frames = []

    s = env.reset()
    done = False

    while not done:
       
        a = agent.act(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0).to(device))
        s_next, _, done, _ = env.step(a)
        s = s_next
        frame = plt.imshow(env.render(mode="rgb_array"), animated=True)
        frames.append([frame])

    animate = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    animate.save(fpath, writer="pillow") 
