import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from kdanet.data import build_dataloaders
from kdanet.losses import KDANetLoss
from kdanet.models import KDANet
from kdanet.models.teacher import load_teacher
from kdanet.utils.checkpoint import save_checkpoint
from kdanet.utils.config import load_config
from kdanet.utils.metrics import classification_metrics
from kdanet.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train KDANet.")
    parser.add_argument("--config", default="configs/kdanet.yaml", help="Path to config file.")
    parser.add_argument(
        "--teacher-factory",
        default=None,
        help="Teacher factory in the form 'module.submodule:function_name'.",
    )
    parser.add_argument("--teacher-checkpoint", default=None, help="Path to teacher checkpoint.")
    parser.add_argument("--device", default="cuda", help="Training device.")
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
def evaluate(model, loader, device, config):
    model.eval()
    all_targets = []
    all_predictions = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs["student_logits"].argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return classification_metrics(
        all_targets,
        all_predictions,
        num_classes=int(config["data"]["num_classes"]),
        class_names=config["data"].get("class_names"),
    )


def train_one_epoch(model, teacher, criterion, loader, optimizer, device, epoch, print_freq):
    model.train()
    running = {"total": 0.0, "classification": 0.0, "contrastive": 0.0, "relational": 0.0, "feature": 0.0}
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)
    for step, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images, teacher=teacher)
        losses = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        for key in running:
            running[key] += float(losses[key].detach()) * batch_size

        if step % print_freq == 0:
            progress.set_postfix({key: running[key] / total_samples for key in running})

    return {key: running[key] / max(total_samples, 1) for key in running}


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_loader, val_loader, _ = build_dataloaders(config)
    model = build_model(config).to(device)

    teacher = None
    if args.teacher_factory:
        teacher = load_teacher(args.teacher_factory, args.teacher_checkpoint, device=device)
    elif args.teacher_checkpoint:
        raise ValueError("--teacher-checkpoint requires --teacher-factory so the teacher architecture can be built.")
    else:
        print("No teacher model was provided. Training will use classification loss only.")

    loss_cfg = config["loss"]
    criterion = KDANetLoss(
        contrastive_weight=float(loss_cfg.get("contrastive_weight", 0.10)),
        relational_weight=float(loss_cfg.get("relational_weight", 0.10)),
        feature_weight=float(loss_cfg.get("feature_weight", 0.10)),
        classification_weight=float(loss_cfg.get("classification_weight", 1.0)),
        contrastive_temperature=float(loss_cfg.get("contrastive_temperature", 0.07)),
    )

    train_cfg = config["train"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("lr_step_size", 60)),
        gamma=float(train_cfg.get("lr_gamma", 0.1)),
    )

    output_dir = Path(train_cfg.get("output_dir", "runs/kdanet"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_oa = 0.0

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_losses = train_one_epoch(
            model,
            teacher,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            print_freq=int(train_cfg.get("print_freq", 20)),
        )
        scheduler.step()

        metrics = evaluate(model, val_loader, device, config)
        oa = metrics["oa"]
        print(
            f"Epoch {epoch}: "
            f"loss={train_losses['total']:.4f}, "
            f"val_OA={oa * 100:.2f}, "
            f"val_AA={metrics['aa'] * 100:.2f}, "
            f"val_Kappa={metrics['kappa'] * 100:.2f}"
        )

        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, best_oa)
        if oa > best_oa:
            best_oa = oa
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, best_oa)
            print(f"Saved best checkpoint with OA={best_oa * 100:.2f}.")


if __name__ == "__main__":
    main()
