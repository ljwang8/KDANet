from typing import Dict

import torch
from torch import nn
from torchvision import models


_WIDTH_TO_FACTORY = {
    "x0_5": models.shufflenet_v2_x0_5,
    "x1_0": models.shufflenet_v2_x1_0,
    "x1_5": models.shufflenet_v2_x1_5,
    "x2_0": models.shufflenet_v2_x2_0,
}


class ShuffleNetV2Student(nn.Module):
    """ShuffleNetV2 student network returning logits and high-level features."""

    def __init__(self, num_classes: int = 9, width: str = "x0_5"):
        super().__init__()
        if width not in _WIDTH_TO_FACTORY:
            valid = ", ".join(sorted(_WIDTH_TO_FACTORY))
            raise ValueError(f"Unsupported ShuffleNetV2 width '{width}'. Valid options: {valid}.")

        try:
            backbone = _WIDTH_TO_FACTORY[width](weights=None)
        except TypeError:
            backbone = _WIDTH_TO_FACTORY[width](pretrained=False)
        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.conv5 = backbone.conv5
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)
        self.feature_channels = backbone.fc.in_features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        features = self.conv5(x)
        embedding = self.pool(features).flatten(1)
        logits = self.classifier(embedding)
        return {"logits": logits, "features": features, "embedding": embedding}
