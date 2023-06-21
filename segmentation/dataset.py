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
    label = Image.open(label_name)

    sample = {
      "image": transforms.ToTensor()(image),
      "label": (255 * transforms.ToTensor()(label)).long()
    }
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


# Snippet to visualise three batches:
#
# ROOT = ...
# transformed_dataset = SMBDataset(folder=ROOT / Path("export"), transform=crop_flip)
# dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)
#
# f, axes = plt.subplots(nrows=3, ncols=8, figsize=(12, 5))
# axes = np.ravel(axes)
#
# for num_batch, batch in enumerate(dataloader):
#   print(f"Batch {num_batch}, images {batch['image'].size()}, labels {batch['label'].size()}")
#   for j, (img, lbl) in enumerate(zip(batch["image"], batch["label"])):
#     ax_img, ax_label = axes[2 * j + num_batch * 8], axes[2 * j + 1 + num_batch * 8]
#     ax_img.imshow(img.permute(1, 2, 0));
#     ax_label.imshow(lbl.permute(1, 2, 0));
#   if num_batch == 2:
#     break
