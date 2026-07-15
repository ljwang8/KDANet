from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_model_checkpoint(path, model, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=False)
    return state
