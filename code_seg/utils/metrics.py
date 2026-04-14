import numpy as np
import torch


EPS = 1e-15


def binary_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = float((y_true * y_pred).sum())
    return (2.0 * intersection + EPS) / (float(y_true.sum()) + float(y_pred.sum()) + EPS)


def binary_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = float((y_true * y_pred).sum())
    union = float(y_true.sum()) + float(y_pred.sum()) - intersection
    return (intersection + EPS) / (union + EPS)


def multiclass_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_background: bool = True,
) -> float:
    """
    Compute mean Dice across classes appearing in y_true.
    This is consistent with the evaluation logic used in the original code.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if y_true.sum() == 0:
        return 1.0 if y_pred.sum() == 0 else 0.0

    scores = []
    class_ids = np.unique(y_true)

    for class_id in class_ids:
        if ignore_background and class_id == 0:
            continue
        gt_mask = (y_true == class_id).astype(np.uint8)
        pred_mask = (y_pred == class_id).astype(np.uint8)
        scores.append(binary_dice(gt_mask, pred_mask))

    if len(scores) == 0:
        return 1.0 if y_pred.sum() == 0 else 0.0
    return float(np.mean(scores))


def multiclass_jaccard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_background: bool = True,
) -> float:
    """
    Compute mean Jaccard across classes appearing in y_true.
    This is consistent with the evaluation logic used in the original code.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if y_true.sum() == 0:
        return 1.0 if y_pred.sum() == 0 else 0.0

    scores = []
    class_ids = np.unique(y_true)

    for class_id in class_ids:
        if ignore_background and class_id == 0:
            continue
        gt_mask = (y_true == class_id).astype(np.uint8)
        pred_mask = (y_pred == class_id).astype(np.uint8)
        scores.append(binary_jaccard(gt_mask, pred_mask))

    if len(scores) == 0:
        return 1.0 if y_pred.sum() == 0 else 0.0
    return float(np.mean(scores))


def evaluate_case(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "dice": multiclass_dice(y_true, y_pred),
        "jaccard": multiclass_jaccard(y_true, y_pred),
    }


class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.dice_scores = []
        self.jaccard_scores = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        case_metrics = evaluate_case(y_true, y_pred)
        self.dice_scores.append(case_metrics["dice"])
        self.jaccard_scores.append(case_metrics["jaccard"])

    def compute(self) -> dict:
        if len(self.dice_scores) == 0:
            return {"dice": 0.0, "jaccard": 0.0}

        return {
            "dice": float(np.mean(self.dice_scores)),
            "jaccard": float(np.mean(self.jaccard_scores)),
        }


def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits [B, C, H, W] to discrete segmentation [B, H, W].
    """
    return torch.argmax(torch.softmax(logits, dim=1), dim=1)