"""Model builders."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet34(num_classes: int) -> nn.Module:
    try:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
