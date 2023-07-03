from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn


class CategoricalCNN(nn.Module):
    """
    Simple CNN taking as input a frame (or stack of frames) and returning
    a distribution over the possible actions.
    """

    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int) -> None:
        super().__init__()
        # Copy the same structure as Mnih
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        # )
        
        # Perhaps too simple for Mario, whose screen changes & introduces new elements? (enemies, decor, ...)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape) -> int:
        """Util to get output shape at the end of the CNN"""
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

    def forward(self, x) -> torch.Tensor:
        """Forward pass through network"""
        conv_out = self.conv(x).view(x.size()[0], -1)  # flatten
        return self.fc(conv_out)


class CategoricalMLP(nn.Module):
    """
    Simple MLP taking as input a 1D-array (or stack of 1D-arrays) and returning
    a distribution over the possible actions.
    """

    def __init__(
        self,
        input_shape: int,
        n_actions: int,
        hidden_sizes: Union[int, List[int]] = [32],
    ):
        super().__init__()

        # Unpack arrays of dimension 1
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            input_shape = input_shape[0]
        if isinstance(n_actions, (list, tuple)) and len(n_actions) == 1:
            n_actions = n_actions[0]

        if isinstance(input_shape, (list, tuple)) or isinstance(n_actions, (list, tuple)):
            e = (
                f"CategoricalMLP can only take one-dimensional input and output shapes "
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
            activation = nn.LeakyReLU if i < len(sizes) - 2 else nn.Identity
            layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """Forward pass through network"""
        return self.mlp(x)
