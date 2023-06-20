import pathlib
from pathlib import Path
import random
from typing import *

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class SMBDataset(Dataset):
  """
  Custom torch Dataset for loading Super Mario Bros. data
  for the segmentation model
  """

  def __init__(self, folder: pathlib.Path, transform: Optional[Callable] = None):
    """
    Args:
      folder (pathlib): path to folder containing folders 'images' and 'labels'
      transform (callable, optional): optional transformation to apply to smaple
    """
    self.folder = folder
    self.transform = transform

  def __len__(self):
    return len(list((self.folder / Path("images")).glob("*.png")))
  
  def __getitem__(self, idx: Union[int, torch.Tensor]):
    """
    Returns dictionary with image and label.
    
    Implementation is dependent on the folder structure and naming convention
    for images and labels (here, simply numerals).
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    image_name = self.folder / Path(f"images/{idx}.png")
    label_name = self.folder / Path(f"labels/{idx}.png")
    # ^ check this works with torch.Tensor
    image = Image.open(image_name).convert("RGB")
    label = Image.open(label_name) # TODO: multiply by 255

    sample = {"image": image, "label": label}
    if self.transform:
      # TODO: might need different transformation for train and eval
      sample = self.transform(sample)
    
    return sample


def crop_flip(sample: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
  """
  Applies random cropping and horizontal flips to train images
  """
  
  # Cropping
  i, j, h, w = transforms.RandomCrop.get_params(sample["image"], (224, 224))
  sample["image"] = TF.crop(sample["image"], i, j, h, w)
  sample["label"] = TF.crop(sample["label"], i, j, h, w)

  # Flipping
  if random.random() > 0.5:
    sample["image"] = TF.hflip(sample["image"])
    sample["label"] = TF.hflip(sample["label"])

  return sample