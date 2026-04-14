import random
from typing import List, Tuple

import torch
from einops import rearrange


def _check_patch_compatibility(tensor: torch.Tensor, patch_size: int) -> None:
    _, _, height, width = tensor.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Input spatial size ({height}, {width}) must be divisible by patch_size={patch_size}."
        )


def _volume_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    return rearrange(
        x,
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=patch_size,
        p2=patch_size,
    )


def _mask_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    return rearrange(
        x,
        "b (h p1) (w p2) -> b (h w) (p1 p2)",
        p1=patch_size,
        p2=patch_size,
    )


def _patches_to_volume(patches: torch.Tensor, batch_size: int, channels: int, height: int, width: int, patch_size: int) -> torch.Tensor:
    return rearrange(
        patches,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        b=batch_size,
        c=channels,
        h=height // patch_size,
        w=width // patch_size,
        p1=patch_size,
        p2=patch_size,
    )


def _patches_to_mask(patches: torch.Tensor, batch_size: int, height: int, width: int, patch_size: int) -> torch.Tensor:
    return rearrange(
        patches,
        "b (h w) (p1 p2) -> b (h p1) (w p2)",
        b=batch_size,
        h=height // patch_size,
        w=width // patch_size,
        p1=patch_size,
        p2=patch_size,
    )


def _sample_patch_indices(candidate_indices: List[int], total_patches: int, num_patches: int) -> List[int]:
    if len(candidate_indices) >= num_patches:
        return random.sample(candidate_indices, num_patches)

    remaining = num_patches - len(candidate_indices)
    all_indices = list(range(total_patches))
    additional = random.sample(all_indices, remaining)
    return candidate_indices + additional


def replace_divergent_patches_with_reference(
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    source_divergence_masks: torch.Tensor,
    reference_images: torch.Tensor,
    reference_labels: torch.Tensor,
    num_patches: int = 2,
    patch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replace patches in the unlabeled source sample using reference labeled patches.

    Patch selection rule:
        preferentially select patches that contain divergence regions
        (i.e. any pixel in the divergence mask equals 1)

    After replacement:
        replaced patches are marked as consensus (set to 0) in the updated divergence mask.

    Returns:
        mixed_images: [B, C, H, W]
        mixed_labels: [B, H, W]
        updated_divergence_masks: [B, H, W]
    """
    _check_patch_compatibility(source_images, patch_size)

    batch_size, channels, height, width = source_images.shape

    image_patches = _volume_to_patches(source_images, patch_size)
    label_patches = _mask_to_patches(source_labels, patch_size)
    divergence_patches = _mask_to_patches(source_divergence_masks.float(), patch_size)

    reference_image_patches = _volume_to_patches(reference_images, patch_size)
    reference_label_patches = _mask_to_patches(reference_labels, patch_size)

    patch_has_divergence = divergence_patches.max(dim=-1).values > 0

    mixed_image_patches = image_patches.clone()
    mixed_label_patches = label_patches.clone()
    updated_divergence_patches = divergence_patches.clone()

    total_patches = patch_has_divergence.shape[1]

    for batch_idx in range(batch_size):
        divergent_patch_indices = torch.where(patch_has_divergence[batch_idx])[0].tolist()
        selected_indices = _sample_patch_indices(
            candidate_indices=divergent_patch_indices,
            total_patches=total_patches,
            num_patches=num_patches,
        )

        for patch_idx in selected_indices:
            mixed_image_patches[batch_idx, patch_idx] = reference_image_patches[batch_idx, patch_idx]
            mixed_label_patches[batch_idx, patch_idx] = reference_label_patches[batch_idx, patch_idx]
            updated_divergence_patches[batch_idx, patch_idx] = 0

    mixed_images = _patches_to_volume(
        mixed_image_patches, batch_size, channels, height, width, patch_size
    )
    mixed_labels = _patches_to_mask(
        mixed_label_patches, batch_size, height, width, patch_size
    )
    updated_divergence_masks = _patches_to_mask(
        updated_divergence_patches, batch_size, height, width, patch_size
    )

    return mixed_images, mixed_labels, updated_divergence_masks


def replace_consensus_patches_into_reference(
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    source_divergence_masks: torch.Tensor,
    reference_images: torch.Tensor,
    reference_labels: torch.Tensor,
    num_patches: int = 5,
    patch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replace consensus patches from the source sample into the labeled reference sample.

    Patch selection rule:
        preferentially select patches that are entirely consensus
        (i.e. max divergence value in the patch equals 0)

    After replacement:
        selected patches are marked as divergence (set to 1) in the new divergence mask.

    Returns:
        mixed_reference_images: [B, C, H, W]
        mixed_reference_labels: [B, H, W]
        new_divergence_masks: [B, H, W]
    """
    _check_patch_compatibility(source_images, patch_size)

    batch_size, channels, height, width = source_images.shape

    source_image_patches = _volume_to_patches(source_images, patch_size)
    source_label_patches = _mask_to_patches(source_labels, patch_size)
    source_divergence_patches = _mask_to_patches(source_divergence_masks.float(), patch_size)

    reference_image_patches = _volume_to_patches(reference_images, patch_size)
    reference_label_patches = _mask_to_patches(reference_labels, patch_size)

    patch_is_consensus = source_divergence_patches.max(dim=-1).values == 0

    mixed_reference_image_patches = reference_image_patches.clone()
    mixed_reference_label_patches = reference_label_patches.clone()
    new_divergence_patches = torch.zeros_like(source_divergence_patches)

    total_patches = patch_is_consensus.shape[1]

    for batch_idx in range(batch_size):
        consensus_patch_indices = torch.where(patch_is_consensus[batch_idx])[0].tolist()
        selected_indices = _sample_patch_indices(
            candidate_indices=consensus_patch_indices,
            total_patches=total_patches,
            num_patches=num_patches,
        )

        for patch_idx in selected_indices:
            mixed_reference_image_patches[batch_idx, patch_idx] = source_image_patches[batch_idx, patch_idx]
            mixed_reference_label_patches[batch_idx, patch_idx] = source_label_patches[batch_idx, patch_idx]
            new_divergence_patches[batch_idx, patch_idx] = 1

    mixed_reference_images = _patches_to_volume(
        mixed_reference_image_patches, batch_size, channels, height, width, patch_size
    )
    mixed_reference_labels = _patches_to_mask(
        mixed_reference_label_patches, batch_size, height, width, patch_size
    )
    new_divergence_masks = _patches_to_mask(
        new_divergence_patches, batch_size, height, width, patch_size
    )

    return mixed_reference_images, mixed_reference_labels, new_divergence_masks


def apply_cpda(
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    source_divergence_masks: torch.Tensor,
    reference_images: torch.Tensor,
    reference_labels: torch.Tensor,
    probability: float = 0.2,
    unlabeled_num_patches: int = 2,
    labeled_num_patches: int = 5,
    patch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if random.random() >= probability:
        return source_images, source_labels, source_divergence_masks

    mixed_unlabeled_images, mixed_unlabeled_labels, mixed_unlabeled_divergence = (
        replace_divergent_patches_with_reference(
            source_images=source_images,
            source_labels=source_labels,
            source_divergence_masks=source_divergence_masks,
            reference_images=reference_images,
            reference_labels=reference_labels,
            num_patches=unlabeled_num_patches,
            patch_size=patch_size,
        )
    )

    mixed_reference_images, mixed_reference_labels, mixed_reference_divergence = (
        replace_consensus_patches_into_reference(
            source_images=source_images,
            source_labels=source_labels,
            source_divergence_masks=source_divergence_masks,
            reference_images=reference_images,
            reference_labels=reference_labels,
            num_patches=labeled_num_patches,
            patch_size=patch_size,
        )
    )

    images = torch.cat([source_images, mixed_unlabeled_images, mixed_reference_images], dim=0)
    pseudo_labels = torch.cat([source_labels, mixed_unlabeled_labels, mixed_reference_labels], dim=0)
    divergence_masks = torch.cat([source_divergence_masks, mixed_unlabeled_divergence, mixed_reference_divergence], dim=0)

    return images, pseudo_labels, divergence_masks