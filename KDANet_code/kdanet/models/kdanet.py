from typing import Dict, Optional

import torch
from torch import nn

from .canlam import CANLAM
from .shufflenetv2 import ShuffleNetV2Student


def _normalize_model_output(output):
    if isinstance(output, dict):
        if "logits" not in output or "features" not in output:
            raise KeyError("Model output dictionary must contain 'logits' and 'features'.")
        return output
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        logits, features = output[:2]
        return {"logits": logits, "features": features}
    raise TypeError("Model output must be a dict or a tuple/list containing logits and features.")


class FeatureProjector(nn.Module):
    """Projects feature maps to the shared distillation space."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class KDANet(nn.Module):
    """KDANet student model with CANLAM-based feature refinement for distillation."""

    def __init__(
        self,
        num_classes: int = 9,
        student_width: str = "x0_5",
        student_feature_channels: int = 1024,
        teacher_feature_channels: int = 1024,
        distill_channels: int = 256,
        canlam_reduction: int = 16,
    ):
        super().__init__()
        self.student = ShuffleNetV2Student(num_classes=num_classes, width=student_width)
        student_feature_channels = self.student.feature_channels or student_feature_channels
        self.student_projector = FeatureProjector(student_feature_channels, distill_channels)
        self.teacher_projector = FeatureProjector(teacher_feature_channels, distill_channels)
        self.canlam = CANLAM(distill_channels, reduction=canlam_reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        x: torch.Tensor,
        teacher: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        student_out = _normalize_model_output(self.student(x))
        student_features = self.student_projector(student_out["features"])
        student_enhanced = self.canlam(student_features)
        student_embedding = self.pool(student_enhanced).flatten(1)

        output = {
            "student_logits": student_out["logits"],
            "student_features": student_features,
            "student_enhanced": student_enhanced,
            "student_embedding": student_embedding,
        }

        if teacher is None:
            return output

        with torch.no_grad():
            teacher_out = _normalize_model_output(teacher(x))

        teacher_features = self.teacher_projector(teacher_out["features"])
        teacher_enhanced = self.canlam(teacher_features)
        teacher_embedding = self.pool(teacher_enhanced).flatten(1)

        output.update(
            {
                "teacher_logits": teacher_out["logits"],
                "teacher_features": teacher_features,
                "teacher_enhanced": teacher_enhanced,
                "teacher_embedding": teacher_embedding,
            }
        )
        return output
