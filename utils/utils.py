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


def get_lr(step: int, decay_steps: int, lr_max: float, lr_min: float):
    """
    Hand-crafted scheduler: decays linearly until lr_min over decay_steps
    """
    return max(lr_min, lr_max - (lr_max - lr_min) / decay_steps * step)


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


def render_in_jupyter(
    env, img: mpl.image.AxesImage = None, info: str = ""
) -> mpl.image.AxesImage:
    """ """
    if img is None:
        # Create first image, once
        img = plt.imshow(env.render(mode="rgb_array"))
        img.axes.set_title(info)
        return img

    # Update existing screen
    img.set_data(env.render(mode="rgb_array"))  # just update the data
    img.axes.set_title(info)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    return img


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
