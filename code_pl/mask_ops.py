from typing import Dict, Tuple

import numpy as np
import torch

from image_io import load_image


def encode_prompt_mask(mask_path: str) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Load a prompt mask and convert it into one-hot object masks.

    Important:
        This function intentionally preserves the original behavior:
        all unique labels in the mask are encoded, including background
        if it is present in the mask.

    Args:
        mask_path: Path to the prompt mask image.

    Returns:
        one_hot_masks:
            Array of shape (num_objects, H, W), dtype uint8.
        class_mapping:
            Mapping from internal index -> original label value.
    """
    mask_array = load_image(mask_path, mode="L")
    unique_labels = np.unique(mask_array)

    class_mapping = {
        idx: int(label_value)
        for idx, label_value in enumerate(unique_labels)
    }

    num_objects = len(unique_labels)
    height, width = mask_array.shape
    one_hot_masks = np.zeros((num_objects, height, width), dtype=np.uint8)

    for idx, label_value in class_mapping.items():
        one_hot_masks[idx, :, :] = (mask_array == label_value)

    return one_hot_masks, class_mapping


def decode_mask_logits(mask_logits: torch.Tensor) -> np.ndarray:
    """
    Decode SAM2 mask logits into an internal index mask.

    Important:
        This function intentionally preserves the original behavior:
        - if logits have shape (N, 1, H, W), squeeze channel dim
        - select argmax along object dimension
        - only pixels with max logit > 0 are considered valid
        - invalid pixels are assigned index 0

    Args:
        mask_logits: Tensor of shape (N, H, W) or (N, 1, H, W).

    Returns:
        NumPy array of shape (H, W), containing internal object indices.
    """
    if mask_logits.dim() == 4:
        mask_logits = mask_logits.squeeze(1)

    max_values, max_indices = torch.max(mask_logits, dim=0)
    output_mask = torch.zeros_like(max_indices, dtype=torch.long)

    valid_mask = max_values > 0
    output_mask[valid_mask] = max_indices[valid_mask]

    return output_mask.cpu().numpy()


def remap_mask_labels(index_mask: np.ndarray, label_mapping: Dict[int, int]) -> np.ndarray:
    """
    Remap an internal index mask back to original semantic label values.

    Args:
        index_mask: Internal index mask of shape (H, W).
        label_mapping: Mapping from internal index -> original label value.

    Returns:
        Semantic label mask of shape (H, W).
    """
    output_mask = np.zeros_like(index_mask)

    for internal_index, original_label in label_mapping.items():
        output_mask[index_mask == internal_index] = original_label

    return output_mask


def build_divergence_mask(
    mask_original: np.ndarray,
    mask_rotated: np.ndarray,
    mask_flipped: np.ndarray,
) -> np.ndarray:
    """
    Build the divergence mask from three view-aligned pseudo labels.

    Definition is intentionally kept identical to the original implementation:
    - consensus across all three views -> 0
    - divergence among views -> 255

    Args:
        mask_original: Pseudo label from original view, aligned to original space.
        mask_rotated: Pseudo label from rotated view, inverse-transformed back.
        mask_flipped: Pseudo label from flipped view, inverse-transformed back.

    Returns:
        Divergence mask of shape (H, W), dtype uint8.
    """
    if mask_original.shape != mask_rotated.shape or mask_original.shape != mask_flipped.shape:
        raise ValueError(
            "All input masks must have the same shape, but got "
            f"{mask_original.shape}, {mask_rotated.shape}, {mask_flipped.shape}"
        )

    is_consistent = (mask_original == mask_rotated) & (mask_original == mask_flipped)
    divergence_mask = np.where(is_consistent, 0, 255).astype(np.uint8)
    return divergence_mask