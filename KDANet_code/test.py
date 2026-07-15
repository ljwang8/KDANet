import argparse

import torch

from kdanet.data import build_dataloaders
from kdanet.models import KDANet
from kdanet.utils.checkpoint import load_model_checkpoint
from kdanet.utils.config import load_config
from kdanet.utils.metrics import classification_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KDANet.")
    parser.add_argument("--config", default="configs/kdanet.yaml", help="Path to config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained KDANet checkpoint.")
    parser.add_argument("--device", default="cuda", help="Evaluation device.")
    return parser.parse_args()


def build_model(config):
    model_cfg = config["model"]
    data_cfg = config["data"]
    return KDANet(
        num_classes=int(data_cfg["num_classes"]),
        student_width=model_cfg.get("student_width", "x0_5"),
        student_feature_channels=int(model_cfg.get("student_feature_channels", 1024)),
        teacher_feature_channels=int(model_cfg.get("teacher_feature_channels", 1024)),
        distill_channels=int(model_cfg.get("distill_channels", 256)),
        canlam_reduction=int(model_cfg.get("canlam_reduction", 16)),
    )


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    _, _, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    load_model_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    all_targets = []
    all_predictions = []
    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs["student_logits"].argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = classification_metrics(
        all_targets,
        all_predictions,
        num_classes=int(config["data"]["num_classes"]),
        class_names=config["data"].get("class_names"),
    )

    print(f"OA: {metrics['oa'] * 100:.2f}")
    print(f"AA: {metrics['aa'] * 100:.2f}")
    print(f"Kappa: {metrics['kappa'] * 100:.2f}")
    print()
    print("Per-class metrics:")
    for row in metrics["per_class"]:
        print(
            f"{row['class']}: "
            f"precision={row['precision'] * 100:.2f}, "
            f"recall={row['recall'] * 100:.2f}, "
            f"f1={row['f1'] * 100:.2f}, "
            f"support={row['support']}"
        )


if __name__ == "__main__":
    main()
