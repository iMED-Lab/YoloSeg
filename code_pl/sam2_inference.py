import gc
import os
from typing import Sequence

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

from image_io import list_image_files, save_image
from mask_ops import encode_prompt_mask
from transforms import invert_transform, validate_view


def build_video_predictor(config_path: str, checkpoint_path: str, device: torch.device):
    """
    Build the SAM2 video predictor.

    Args:
        config_path: Path to the SAM2 config file.
        checkpoint_path: Path to the SAM2 checkpoint.
        device: Torch device.

    Returns:
        Predictor instance created by build_sam2_video_predictor.
    """
    return build_sam2_video_predictor(config_path, checkpoint_path, device=device)


def _find_prompt_frame_index(frame_names, prompt_frame_name: str) -> int:
    """
    Find the index of the prompt frame in the sorted frame sequence.

    Preserves original behavior:
    - if prompt_frame_name exists in frame_names, use its index
    - otherwise fall back to index 0
    """
    if prompt_frame_name in frame_names:
        return frame_names.index(prompt_frame_name)
    return 0


def _add_prompt_masks_to_state(
    predictor,
    inference_state,
    one_hot_masks: np.ndarray,
    class_mapping,
    ann_frame_idx: int,
) -> None:
    """
    Add all prompt masks to the predictor state for the anchor frame.
    """
    num_objects = one_hot_masks.shape[0]

    for internal_index in range(num_objects):
        object_id = class_mapping[internal_index]
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=one_hot_masks[internal_index],
        )


def _decode_mask_logits_with_obj_ids(
    mask_logits: torch.Tensor,
    out_obj_ids: Sequence[int],
) -> np.ndarray:
    """
    Decode SAM2 mask logits into a semantic label mask using out_obj_ids.

    This is the key fix for multi-class robustness:
    we no longer assume that the channel index order in `mask_logits`
    is identical to the original internal class order. Instead, we use
    `out_obj_ids` returned by SAM2 to explicitly map each channel to its
    semantic label value.

    Args:
        mask_logits: Tensor of shape (N, H, W) or (N, 1, H, W).
        out_obj_ids: Sequence of object ids corresponding to the channel order
            in `mask_logits`. In this project, obj_id is the original label value.

    Returns:
        Semantic label mask of shape (H, W), dtype uint8-compatible integer array.
    """
    if mask_logits.dim() == 4:
        mask_logits = mask_logits.squeeze(1)

    if mask_logits.dim() != 3:
        raise ValueError(
            f"Expected mask_logits to have 3 dims after squeeze, got shape {tuple(mask_logits.shape)}"
        )

    num_channels = mask_logits.shape[0]
    if len(out_obj_ids) != num_channels:
        raise ValueError(
            "Length of out_obj_ids does not match logits channels: "
            f"{len(out_obj_ids)} vs {num_channels}"
        )

    max_values, max_indices = torch.max(mask_logits, dim=0)

    semantic_mask = torch.zeros_like(max_indices, dtype=torch.long)
    valid_mask = max_values > 0

    if valid_mask.any():
        obj_id_tensor = torch.as_tensor(
            list(out_obj_ids),
            device=max_indices.device,
            dtype=torch.long,
        )
        semantic_mask[valid_mask] = obj_id_tensor[max_indices[valid_mask]]

    return semantic_mask.cpu().numpy()


def _save_prediction_mask(
    mask_logits: torch.Tensor,
    out_obj_ids: Sequence[int],
    output_path: str,
    view: str,
) -> None:
    """
    Decode, inverse-transform, and save the predicted mask.
    """
    pred_label_mask = _decode_mask_logits_with_obj_ids(mask_logits, out_obj_ids)
    corrected_mask = invert_transform(pred_label_mask, view)
    save_image(corrected_mask, output_path, mode="L")


def _propagate_one_direction(
    predictor,
    inference_state,
    one_hot_masks: np.ndarray,
    class_mapping,
    ann_frame_idx: int,
    frame_names,
    output_dir: str,
    view: str,
    reverse: bool,
) -> None:
    """
    Run propagation in one direction and save all predicted masks.
    """
    if reverse:
        predictor.reset_state(inference_state)

    _add_prompt_masks_to_state(
        predictor=predictor,
        inference_state=inference_state,
        one_hot_masks=one_hot_masks,
        class_mapping=class_mapping,
        ann_frame_idx=ann_frame_idx,
    )

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        reverse=reverse,
    ):
        output_name = frame_names[out_frame_idx]
        output_path = os.path.join(output_dir, output_name)

        _save_prediction_mask(
            mask_logits=out_mask_logits,
            out_obj_ids=out_obj_ids,
            output_path=output_path,
            view=view,
        )


def run_bidirectional_propagation(
    predictor,
    frame_dir: str,
    prompt_frame_name: str,
    prompt_mask_path: str,
    output_dir: str,
    view: str,
) -> None:
    """
    Run single-view bidirectional propagation and save pseudo labels.

    This function preserves the original algorithmic behavior:
    1. Initialize a SAM2 inference state from the frame directory
    2. Use the selected prompt frame as anchor
    3. Add one-hot prompt masks as object prompts
    4. Run forward propagation
    5. If anchor index > 0, run reverse propagation after resetting state
    6. For each predicted frame:
       decode logits -> inverse transform -> save

    Args:
        predictor: SAM2 video predictor.
        frame_dir: Directory containing frame sequence for this view.
        prompt_frame_name: Prompt frame filename.
        prompt_mask_path: Prompt mask path for this view.
        output_dir: Output directory for pseudo labels of this view.
        view: One of {"original", "rotate", "flip"}.
    """
    validate_view(view)
    os.makedirs(output_dir, exist_ok=True)

    frame_names = list_image_files(frame_dir)
    if len(frame_names) == 0:
        raise ValueError(f"No image files found in frame directory: {frame_dir}")

    one_hot_masks, class_mapping = encode_prompt_mask(prompt_mask_path)
    ann_frame_idx = _find_prompt_frame_index(frame_names, prompt_frame_name)

    inference_state = predictor.init_state(frame_dir, offload_video_to_cpu=True)
    predictor.reset_state(inference_state)

    try:
        _propagate_one_direction(
            predictor=predictor,
            inference_state=inference_state,
            one_hot_masks=one_hot_masks,
            class_mapping=class_mapping,
            ann_frame_idx=ann_frame_idx,
            frame_names=frame_names,
            output_dir=output_dir,
            view=view,
            reverse=False,
        )

        if ann_frame_idx > 0:
            _propagate_one_direction(
                predictor=predictor,
                inference_state=inference_state,
                one_hot_masks=one_hot_masks,
                class_mapping=class_mapping,
                ann_frame_idx=ann_frame_idx,
                frame_names=frame_names,
                output_dir=output_dir,
                view=view,
                reverse=True,
            )
    finally:
        del inference_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()