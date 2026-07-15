from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def contrastive_distillation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    student_embedding = F.normalize(student_embedding, dim=1)
    teacher_embedding = F.normalize(teacher_embedding.detach(), dim=1)
    logits = torch.matmul(student_embedding, teacher_embedding.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def relational_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    student_prob = F.softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach(), dim=1)
    student_prob = F.normalize(student_prob, dim=1)
    teacher_prob = F.normalize(teacher_prob, dim=1)
    student_relation = torch.matmul(student_prob, student_prob.t())
    teacher_relation = torch.matmul(teacher_prob, teacher_prob.t())
    return F.mse_loss(student_relation, teacher_relation)


def feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    if student_features.shape[-2:] != teacher_features.shape[-2:]:
        teacher_features = F.interpolate(
            teacher_features,
            size=student_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return F.mse_loss(student_features, teacher_features.detach())


class KDANetLoss(nn.Module):
    """Composite KDANet objective."""

    def __init__(
        self,
        contrastive_weight: float = 0.10,
        relational_weight: float = 0.10,
        feature_weight: float = 0.10,
        classification_weight: float = 1.0,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.relational_weight = relational_weight
        self.feature_weight = feature_weight
        self.classification_weight = classification_weight
        self.contrastive_temperature = contrastive_temperature
        self.classification = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_loss = self.classification(outputs["student_logits"], targets)
        total = self.classification_weight * cls_loss
        losses = {
            "classification": cls_loss,
            "total": total,
        }

        has_teacher = {
            "teacher_logits",
            "teacher_enhanced",
            "teacher_embedding",
        }.issubset(outputs)

        if not has_teacher:
            losses.update(
                {
                    "contrastive": torch.zeros_like(cls_loss),
                    "relational": torch.zeros_like(cls_loss),
                    "feature": torch.zeros_like(cls_loss),
                }
            )
            return losses

        contrastive = contrastive_distillation_loss(
            outputs["student_embedding"],
            outputs["teacher_embedding"],
            temperature=self.contrastive_temperature,
        )
        relational = relational_distillation_loss(
            outputs["student_logits"],
            outputs["teacher_logits"],
        )
        feature = feature_distillation_loss(
            outputs["student_enhanced"],
            outputs["teacher_enhanced"],
        )

        total = (
            self.classification_weight * cls_loss
            + self.contrastive_weight * contrastive
            + self.relational_weight * relational
            + self.feature_weight * feature
        )
        losses.update(
            {
                "contrastive": contrastive,
                "relational": relational,
                "feature": feature,
                "total": total,
            }
        )
        return losses
