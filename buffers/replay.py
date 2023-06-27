import logging
import numpy as np
import numpy.typing as npt
from pathlib import Path
import random
import torch
from torch.utils.data.dataset import IterableDataset
from typing import *

from .sum_tree import SumTree

logger = logging.getLogger("replay-buffer")


class ReplayBuffer:
    """
    Stores (S, A, R, S', d) transitions for a Q-learning agent,
    which are randomly sampled for experience replay.
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
            raise NotImplementedError("Need to load from parquet from numpy")
            # Check https://stackoverflow.com/questions/57683276/how-to-convert-numpy-to-parquet-without-using-pandas 
            self.states = torch.load(save_dir / Path("states.pt"))
            self.states_next = torch.load(save_dir / Path("states_next.pt"))
            self.actions = torch.load(save_dir / Path("actions.pt"))
            self.rewards = torch.load(save_dir / Path("rewards.pt"))
            self.dones = torch.load(save_dir / Path("dones.pt"))
            self.memory_num_experiences = self.states.shape[0]
            logger.info(f"Restored buffer with size {memory_size} from {str(save_dir)}")
        else:
            self.states = np.zeros((memory_size, *state_space), dtype=np.float32)
            self.states_next = np.zeros((memory_size, *state_space), dtype=np.float32)
            self.actions = np.zeros((memory_size,), dtype=np.int8)
            self.rewards = np.zeros((memory_size,), dtype=np.float32)
            self.dones = np.zeros((memory_size,), dtype=np.int8)
            logger.info(f"Created buffer with size {memory_size}")
    

    def store(self, s: npt.NDArray[np.float32], a: int, r: float, s_next: npt.NDArray[np.float32], d: int) -> None:
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
    

    def sample(
            self, batch_size: int, replace: bool, device: str, return_indices: bool = False
        ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], npt.NDArray]]:
        """ Samples a batch of transitions uniformly from the buffer """

        # Sample `batch_size` transitions at random for experience replay
        idcs = np.random.choice(self.memory_num_experiences, batch_size, replace=replace)

        data = {
            "s": (self.states[idcs], torch.float32),
            "a": (self.actions[idcs], torch.long), # needed for `.gather`
            "r": (self.rewards[idcs], torch.float32),
            "s_next": (self.states_next[idcs], torch.float32),
            "d": (self.dones[idcs], torch.uint8),
        }
        # Convert to tensors
        data = {k: torch.as_tensor(v, dtype=_dtype).to(device) for k, (v, _dtype) in data.items()}
        if return_indices:
            return data, idcs
        return data


    def save(self) -> None:
        """ Saves buffer to disc """
        raise NotImplementedError("Need to store to parquet from numpy")
        # Check https://stackoverflow.com/questions/57683276/how-to-convert-numpy-to-parquet-without-using-pandas
        torch.save(self.states, self.save_dir / Path("states.pt"))
        torch.save(self.states_next, self.save_dir / Path("states_next.pt"))
        torch.save(self.actions, self.save_dir / Path("actions.pt"))
        torch.save(self.rewards, self.save_dir / Path("rewards.pt"))
        torch.save(self.dones, self.save_dir / Path("dones.pt"))
        logger.log(f"Successfully saved buffer to {str(self.save_dir)}")



class PrioritisedReplayBuffer(ReplayBuffer):
    """
    Stores (S, A, R, S', d) transitions for a Q-learning agent,
    which are sampled using Prioritised Experience Replay.

    For full details, see Prioritized Experienced Replay (Schaul et al., 2016)
    """
    def __init__(self, alpha, beta, p_min, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)

        self.tree = SumTree(memory_size=self.memory_size)
        # ^ sum tree data structure to sample from probability intervals

        # Additional parameters for prioritised replay
        self.alpha = alpha # controls priorisation
        self.beta = beta # controls importance-sampling correction
        self.p_min = p_min # to avoid zero prob.
        self.p_max = p_min # for new samples
    

    def store(self, s: npt.NDArray[np.float32], a: int, r: float, s_next: npt.NDArray[np.float32], d: int) -> None:
        """ Stores transition (S, A, R, S', d) in buffer """
        # Add transition (its index) to the tree with default probability
        self.tree.add(self.p_max, self.memory_pointer)
        # Add transition to buffer, as usual
        super().store(s, a, r, s_next, d)


    def sample(
        self, batch_size: int, device: str
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
        """
        Samples a batch of transitions from the buffer using prioritised
        experience replay.

        Note: for simplicity, this method doesn't implement importance-sampling
        correction.
        """

        sample_idcs, tree_idcs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float32)

        # We want a sampling strategy that... TODO
        # Divide the probability range into `batch_size` many segments,
        # then sample one transition at random from each segment

        segment = self.tree.total / batch_size # segment size
        for i in range(batch_size):
            # Sample at random from current segment
            low, high = segment * i, segment * (i + 1)
            cumsum = random.uniform(low, high)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            sample_idcs.append(sample_idx)
            tree_idcs.append(tree_idx)
        
        # P(i) = p_i^α / \sum_{k} p_k^α
        # Will be raised to the αth power in `update_priorities`
        probs = priorities / self.tree.total
        # Importance sampling to correct bias
        weights = (self.memory_num_experiences * probs) ** -self.beta
        weights = weights / weights.max()

        # Format data & return
        data = {
            "s": (self.states[sample_idcs], torch.float32),
            "a": (self.actions[sample_idcs], torch.long), # needed for `.gather`
            "r": (self.rewards[sample_idcs], torch.float32),
            "s_next": (self.states_next[sample_idcs], torch.float32),
            "d": (self.dones[sample_idcs], torch.uint8),
        }
        data = {k: torch.as_tensor(v, dtype=_dtype).to(device) for k, (v, _dtype) in data.items()}
        return data, weights.squeeze().to(device), tree_idcs


    def update_priorities(self, data_idcs: list, td_errors: npt.NDArray[np.float32]):
        """
        Updates priority for each sample as p_i = (|δ_i| + p_min)^α
        based on latest evaluation by Q-network.
        """
        for data_idx, delta in zip(data_idcs, td_errors):
            # Update priority of data samples: p_i = |δ_i| + p_min
            delta = (delta + self.p_min) ** self.alpha
            self.tree.update(data_idx, delta)
            # Update max. priority (given to newly drawn samples)
            self.p_max = max(self.p_max, delta)


class LightningBuffer(IterableDataset):
    """
    TODO: implement as in https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
    """
    def __init__(self, buffer: ReplayBuffer) -> None:
        raise NotImplementedError("Compatibility with lightning not yet implemented")
