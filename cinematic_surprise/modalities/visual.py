"""
modalities/visual.py

ResNet-50 hierarchical visual feature extractor — levels L1 through L4.

Cortical hierarchy mapping (Yamins & DiCarlo 2016; Schrimpf et al. 2020):
    L1 → layer1 → (256,)   V1:  oriented edges, spatial frequencies
    L2 → layer2 → (512,)   V2/V4: textures, curves, color blobs
    L3 → layer3 → (1024,)  LOC: object parts
    L4 → layer4 → (2048,)  IT:  whole objects, faces, scenes

Global Average Pooling collapses spatial dimensions to one vector per stage.

IN:  batch of BGR frames (B, H, W, 3) uint8
OUT: dict mapping channel name → (B, d) float32 numpy array
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn

from cinematic_surprise.config import (
    CNN_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESNET_DEVICE,
    RESNET_LAYER_MAP,
)

try:
    from torchvision.models import ResNet50_Weights
    import torchvision.models as tvm
    _WEIGHTS = ResNet50_Weights.IMAGENET1K_V1
    _NEW_API = True
except ImportError:
    import torchvision.models as tvm
    _NEW_API = False


class VisualExtractor(nn.Module):
    """
    Frozen ResNet-50 with tap-points at four stages.

    Args:
        device : 'cuda' or 'cpu'. Falls back to cpu if cuda unavailable.
    """

    def __init__(self, device: str = RESNET_DEVICE):
        super().__init__()
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained ResNet-50
        if _NEW_API:
            resnet = tvm.resnet50(weights=_WEIGHTS)
        else:
            resnet = tvm.resnet50(pretrained=True)

        resnet.eval()
        for p in resnet.parameters():
            p.requires_grad_(False)

        # Decompose into extractable stages
        self.stem    = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1  = resnet.layer1    # → (B, 256, 56, 56)
        self.layer2  = resnet.layer2    # → (B, 512, 28, 28)
        self.layer3  = resnet.layer3    # → (B,1024, 14, 14)
        self.layer4  = resnet.layer4    # → (B,2048,  7,  7)
        self.avgpool = resnet.avgpool

        # ImageNet normalisation tensors on device
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std",  std)

        self.to(self.device)

    def _preprocess(self, frames_bgr: np.ndarray) -> torch.Tensor:
        """
        (B, H, W, 3) uint8 BGR → (B, 3, 224, 224) float32 normalised tensor.
        """
        batch = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (CNN_INPUT_SIZE, CNN_INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
            batch.append(rgb)

        arr = np.stack(batch).astype(np.float32) / 255.0
        x   = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def extract(self, frames_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract GAP feature vectors for all four ResNet levels.

        Args:
            frames_bgr : (B, H, W, 3) uint8 BGR batch

        Returns:
            Dict: channel name → (B, d) float32 numpy array
                  "L1" → (B, 256)
                  "L2" → (B, 512)
                  "L3" → (B, 1024)
                  "L4" → (B, 2048)
        """
        x  = self._preprocess(frames_bgr)
        x  = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        def gap(t: torch.Tensor) -> np.ndarray:
            # Global average pool: (B, C, H, W) → (B, C)
            return t.mean(dim=[2, 3]).cpu().numpy().astype(np.float32)

        return {
            "L1": gap(f1),
            "L2": gap(f2),
            "L3": gap(f3),
            "L4": gap(f4),
        }
