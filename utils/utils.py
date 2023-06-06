from IPython import display
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from typing import *


def get_n_trainable_params(model: torch.nn.Module) -> int:
    # Get all trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Sum their dimensions
    n_trainable_params = sum(
        np.prod(p.detach().numpy().shape for p in trainable_params)
    )
    return n_trainable_params


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
