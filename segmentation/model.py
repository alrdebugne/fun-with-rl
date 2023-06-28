import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import *

import torch
from torchvision import models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class SegModel(torch.nn.Module):
    """ """
    def __init__(self, path_model: Path, num_classes: int, transform: Optional[Callable] = None):
        """
        Args
            path_model (Path): path with fine-tuned weights
            num_classes (int): number of segmentation labels used in training
            transform (Callable): transformation function based on torchvision.transforms or Compose
        """
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load DeepLabv3 model, adapt classifier and load fine-tuned weights
        segmodel = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
        segmodel.classifier = DeepLabHead(2048, num_classes)
        segmodel.to(device)

        # Load fine-tuned weights
        segmodel.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
        segmodel.eval()  # required because of BatchNorm
        self.segmodel = segmodel
        self.num_classes = num_classes
        self.transform = transform
        self.device = device


    def apply(self, frame: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """ Segments `frame` using fine-tuned segmentation model """
        frame = transforms.ToTensor()(frame).unsqueeze(0).to(self.device)  # adds batch dimension
        if self.transform:
            frame = self.transform(frame)
        logits = self.segmodel(frame)["out"].squeeze(0)
        return torch.argmax(logits, dim=0).detach().cpu().numpy()
