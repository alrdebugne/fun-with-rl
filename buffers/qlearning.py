import logging
import numpy as np
import numpy.typing as npt
from pathlib import Path
import torch
from typing import *

logger = logging.getLogger("q-buffer")


class ReplayBuffer:
    """
    Stores (S, A, R, S', d) transitions for a Q-learning agent,
    with methods for sampling batches of transitions for experience replay.
    
    Sampling can be done either at random or with prioritised sampling (following D4PG).
    (yet to be implemented)
    """

    def __init__(
        self,
        state_space: npt.NDArray[np.float64],
        memory_size: int,
        from_pretrained: bool,
        save_dir: Union[Path, str] = Path("buffer"),
    ) -> None:

        self.memory_size = memory_size
        self.memory_pointer = 0
        self.memory_num_experiences = 0
        self.save_dir = save_dir

        # Create buffers to store transitions
        # Either restore old buffers or create new ones
        if from_pretrained:
            self.states = torch.load(save_dir / Path("states.pt"))
            self.states_next = torch.load(save_dir / Path("states_next.pt"))
            self.actions = torch.load(save_dir / Path("actions.pt"))
            self.rewards = torch.load(save_dir / Path("rewards.pt"))
            self.dones = torch.load(save_dir / Path("dones.pt"))
            self.memory_num_experiences = self.states.shape[0]
            logger.info(f"Restored buffer with size {memory_size} from {str(save_dir)}")
        else:
            self.states = torch.zeros((memory_size, *state_space), dtype=torch.float32)
            self.states_next = torch.zeros((memory_size, *state_space), dtype=torch.float32)
            self.actions = torch.zeros((memory_size,), dtype=torch.int8)
            self.rewards = torch.zeros((memory_size,), dtype=torch.float32)
            self.dones = torch.zeros((memory_size,), dtype=torch.int8)
            logger.info(f"Created buffer with size {memory_size}")
    

    def store(self, s: torch.Tensor, a: int, r: float, s_next: torch.Tensor, d: int) -> None:
        """ Stores transition (S, A, R, S', d) in buffer """
        # Store transitions at current pointer
        self.states[self.memory_pointer] = s
        self.states_next[self.memory_pointer] = s_next
        self.actions[self.memory_pointer] = a
        self.rewards[self.memory_pointer] = r
        self.dones[self.memory_pointer] = d

        # Update pointer
        self.memory_pointer = (self.memory_pointer + 1) % self.memory_size
        self.memory_num_experiences = min(self.memory_num_experiences + 1, self.memory_size)
    

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of transitions from the buffer
        TODO: enable prioritised sampling for multi-world (following D4PG)
        """

        # Sample `batch_size` transitions at random for experience replay
        idcs = np.random.choice(self.memory_num_experiences, batch_size, replace=False)
        # TODO: code prioritised sampling for multi-world following D4PG

        data = {
            "s": self.states[idcs],
            "a": self.actions[idcs],
            "r": self.rewards[idcs],
            "s_next": self.states_next[idcs],
            "d": self.dones[idcs],
        }
        # Convert to tensors
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        return data


    def save(self) -> None:
        """ Saves buffer to disc """
        torch.save(self.states, self.save_dir / Path("states.pt"))
        torch.save(self.states_next, self.save_dir / Path("states_next.pt"))
        torch.save(self.actions, self.save_dir / Path("actions.pt"))
        torch.save(self.rewards, self.save_dir / Path("rewards.pt"))
        torch.save(self.dones, self.save_dir / Path("dones.pt"))
        logger.log(f"Successfully saved buffer to {str(self.save_dir)}")

