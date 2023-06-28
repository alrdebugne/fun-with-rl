import random
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import dataloader


def set_seed(seed: int):
  """ Sets seed everywhere I can think of for reproducibility """
  print(f"Setting seed {seed}")
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def _fast_hist(
    labels_pred: List[float], labels_true: List[float], num_classes: int
) -> npt.NDArray[np.float32]:
  """ """
  mask = (labels_true >= 0) & (labels_true < num_classes)
  hist = np.bincount(
      num_classes * labels_true[mask].astype(int) + labels_pred[mask],
      minlength=num_classes ** 2
  ).reshape(num_classes, num_classes)
  return hist


def get_accuracies_per_class(
    labels_pred: List[float], labels_true: List[float], num_classes: int
) -> List[float]:
  """ Compute class-wise accuracy """
  hist = np.zeros((num_classes, num_classes))
  for labels_pred, labels_true in zip(labels_pred, labels_true):
    hist += _fast_hist(labels_pred.flatten(), labels_true.flatten(), num_classes)

  iou = np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist))
  return iou


def show_segmentation(
    n :int,
    dataloader: dataloader.DataLoader,
    model: torch.nn.Module,
    num_classes: int,
    device: str,
    subplot_kwargs: Optional[Dict] = {}
):
    """ """
    model.eval()
    batch_size = dataloader.batch_size

    # Create figure
    _, axes = plt.subplots(nrows=n * batch_size, ncols=3, **subplot_kwargs)
    axes = np.ravel(axes)
    [a.axis("off") for a in axes]

    for num_batch, batch in enumerate(dataloader):
        # Predict segmentation maps for current batch
        images, labels = batch["image"].to(device), batch["label"].to(device)
        logits = model(images)["out"]
        preds = torch.argmax(logits, 1) # (b, num_classes, h, w)

        # Plot
        for i, pred in enumerate(preds):
            idx = 3 * num_batch * batch_size + 3 * i  # offset for plotting
            axes[idx].imshow(images[i].cpu().permute(1, 2, 0))
            axes[idx + 1].imshow(labels[i].cpu().permute(1, 2, 0), vmin=0, vmax=num_classes, cmap="jet")
            axes[idx + 2].imshow(pred.cpu(), vmin=0, vmax=num_classes, cmap="jet")
        
        if num_batch == n - 1:
            break
