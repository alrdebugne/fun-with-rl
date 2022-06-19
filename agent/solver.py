from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn


class FrameToActionNetwork(nn.Module):
    """
    Simple network taking as input a video frame (or stack of frames) and outputting
    a distribution over the possible action space.
    """

    # TODO: add dropout layers

    def __init__(
        self, input_shape: Tuple[int, int, int], n_actions: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape) -> int:
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

    def forward(self, x) -> None:
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class SimpleMLPNetwork(nn.Module):
    """
    Simple MLP, mainly for testing purposes.
    """

    def __init__(
        self, input_shape: int, n_actions: int, hidden_sizes: Union[int, List[int]]
    ):
        super().__init__()

        if isinstance(input_shape, (list, tuple)) or isinstance(
            n_actions, (list, tuple)
        ):
            e = (
                "SimpleMLPNetowrk can only take one-dimensional input and output shapes "
                f"but got {input_shape} and {n_actions} instead (I/O)."
            )
            raise NotImplementedError(e)

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        # Build fully connected network with dimensions: input_shape -> *hidden_sizes -> n_actions
        # and ReLU activation for hidden layers
        layers = []
        sizes = [input_shape] + hidden_sizes + [n_actions]
        for i in range(len(sizes) - 1):
            activation = nn.ReLU if i < len(sizes) - 2 else nn.Identity
            layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)
