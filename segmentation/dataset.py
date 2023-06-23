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

  def __init__(self, folder, mode: str, transform: Optional[Callable] = None):
    """
    Args:
      folder (pathlib): path to folder containing folders "images" and "labels"
      mode (str): either "train" or "eval". Affects how dataset is partitioned
      transform (callable, optional): optional transformation to apply to sample
    """
    assert mode in ["train", "eval"], \
      f"`mode` must be 'train' or 'eval', but was {mode}."

    self.folder = folder
    self.mode = mode
    self.transform = transform

    # Split images into train and validations
    all_images = sorted(list((self.folder / Path("images")).glob("*.png")))
    all_labels = sorted(list((self.folder / Path("labels")).glob("*.png")))
    # ^ sorting needed to align indices in images and labels

    if mode == "train":
      start, end = 0, int(len(all_images) * 0.8)
    else:
      start, end = int(len(all_images) * 0.8), len(all_images)

    self.images = all_images[start : end]
    self.labels = all_labels[start : end]


  def __len__(self):
    return len(self.images)


  def __getitem__(self, idx: Union[int, torch.Tensor]):
    """
    Returns dictionary with image and label.

    Implementation is dependent on the folder structure and naming convention
    for images and labels (here, simply numerals).
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()

    image_name = self.images[idx]
    label_name = self.labels[idx]
    # TODO: check this works with torch.Tensor
    image = Image.open(image_name).convert("RGB")
    label = Image.open(label_name)

    sample = {
      "image": transforms.ToTensor()(image),
      "label": (255 * transforms.ToTensor()(label)).long()
    }
    if self.transform:
      sample = self.transform(sample, self.mode)

    return sample

def crop_flip(sample: Dict[str, Image.Image], mode: str) -> Dict[str, Image.Image]:
  """
  During training, applies random cropping and horizontal flips
  At evaluation, simply centre-crops images.
  """

  if mode == "train":
    # cropping
    i, j, h, w = transforms.RandomCrop.get_params(sample["image"], (224, 224))
    sample["image"] = TF.crop(sample["image"], i, j, h, w)
    sample["label"] = TF.crop(sample["label"], i, j, h, w)
    # flipping
    if random.random() > 0.5:
      sample["image"] = TF.hflip(sample["image"])
      sample["label"] = TF.hflip(sample["label"])

  else:
    sample["image"] = TF.center_crop(sample["image"], [224, 224])
    sample["label"]  = TF.center_crop(sample["label"], [224, 224])

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
