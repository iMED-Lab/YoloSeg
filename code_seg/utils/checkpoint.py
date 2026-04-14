import os
from typing import Optional

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_model_weights(model: torch.nn.Module, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model_weights(model: torch.nn.Module, path: str, map_location: Optional[str] = None) -> torch.nn.Module:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def save_training_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_training_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_dice = checkpoint.get("best_dice", -1.0)

    return model, optimizer, epoch, best_dice