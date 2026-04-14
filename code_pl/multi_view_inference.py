import argparse
import os
from typing import Dict

import torch
from tqdm import tqdm

from data_manager import (
    TemporaryViewDataset,
    ensure_output_dirs,
    get_output_dir_for_view,
    get_prompt_paths,
    read_single_prompt_filename,
    resolve_stage1_paths,
)
from image_io import list_image_files, load_image, save_image
from mask_ops import build_divergence_mask
from sam2_inference import build_video_predictor, run_bidirectional_propagation
from transforms import (
    VIEW_FLIP,
    VIEW_ORIGINAL,
    VIEW_ROTATE,
    SUPPORTED_VIEWS,
)


def setup_device(device_str: str) -> torch.device:
    """
    Resolve and configure the runtime device.

    Behavior intentionally keeps the original optimization settings:
    - use bfloat16 autocast on CUDA
    - enable TF32 on Ampere+ GPUs when available

    Args:
        device_str: Requested device string, e.g. "cuda" or "cpu".

    Returns:
        torch.device object.
    """
    requested = device_str.lower().strip()

    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        return device

    return torch.device("cpu")


def generate_divergence_masks(
    original_dir: str,
    rotate_dir: str,
    flip_dir: str,
    output_dir: str,
) -> None:
    """
    Generate divergence masks from three view-aligned pseudo-label directories.

    Definition is consistent with the original implementation:
    - all three views agree -> 0
    - otherwise -> 255

    Args:
        original_dir: Directory containing pseudo labels from the original view.
        rotate_dir: Directory containing pseudo labels from the rotated view,
            already inverse-transformed back to original space.
        flip_dir: Directory containing pseudo labels from the flipped view,
            already inverse-transformed back to original space.
        output_dir: Directory to save divergence masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_names = list_image_files(original_dir)
    if len(frame_names) == 0:
        raise ValueError(f"No pseudo-label files found in original view dir: {original_dir}")

    for frame_name in tqdm(frame_names, desc="Generating divergence masks", leave=False):
        original_path = os.path.join(original_dir, frame_name)
        rotate_path = os.path.join(rotate_dir, frame_name)
        flip_path = os.path.join(flip_dir, frame_name)

        if not os.path.isfile(rotate_path):
            raise FileNotFoundError(f"Missing rotated-view pseudo label: {rotate_path}")
        if not os.path.isfile(flip_path):
            raise FileNotFoundError(f"Missing flipped-view pseudo label: {flip_path}")

        mask_original = load_image(original_path, mode="L")
        mask_rotate = load_image(rotate_path, mode="L")
        mask_flip = load_image(flip_path, mode="L")

        divergence_mask = build_divergence_mask(
            mask_original=mask_original,
            mask_rotated=mask_rotate,
            mask_flipped=mask_flip,
        )

        save_path = os.path.join(output_dir, frame_name)
        save_image(divergence_mask, save_path, mode="L")


def run_multi_view_pseudo_label_generation(
    data_root: str,
    checkpoint_path: str,
    config_path: str,
    device_str: str = "cuda",
    overwrite: bool = False,
) -> None:
    """
    Run stage-1 multi-view pseudo-label generation.

    Expected dataset structure:
        data_root/
          file_list/
            train_label_frames.txt
          Train/
            JPEGImages/
            Annotations/

    Outputs will be written to:
        Train/pl_original/
        Train/pl_rotate/
        Train/pl_flip/
        Train/divergence_mask/

    Args:
        data_root: Dataset root directory.
        checkpoint_path: SAM2 checkpoint path.
        config_path: SAM2 config path.
        device_str: Device string, usually "cuda" or "cpu".
        overwrite: Whether to overwrite existing output directories.
    """
    paths = resolve_stage1_paths(data_root=data_root)
    ensure_output_dirs(paths, overwrite=overwrite)

    prompt_frame_name = read_single_prompt_filename(paths.label_list_path)
    _, prompt_mask_path = get_prompt_paths(paths, prompt_frame_name)

    device = setup_device(device_str)
    print(f"Using device: {device}")

    print("Loading SAM2 video predictor...")
    predictor = build_video_predictor(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print("SAM2 video predictor loaded.")

    views = [VIEW_ORIGINAL, VIEW_ROTATE, VIEW_FLIP]

    for view in views:
        output_dir = get_output_dir_for_view(paths, view)
        print(f"Running pseudo-label generation for view: {view}")

        with TemporaryViewDataset(
            frames_dir=paths.frames_dir,
            prompt_mask_path=prompt_mask_path,
            view=view,
        ) as (view_frame_dir, view_prompt_mask_path):
            run_bidirectional_propagation(
                predictor=predictor,
                frame_dir=view_frame_dir,
                prompt_frame_name=prompt_frame_name,
                prompt_mask_path=view_prompt_mask_path,
                output_dir=output_dir,
                view=view,
            )

    print("Generating divergence masks...")
    generate_divergence_masks(
        original_dir=paths.output_original_dir,
        rotate_dir=paths.output_rotate_dir,
        flip_dir=paths.output_flip_dir,
        output_dir=paths.output_divergence_dir,
    )
    print("Stage-1 pseudo-label generation completed.")


def build_argparser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for stage-1 pseudo-label generation.
    """
    parser = argparse.ArgumentParser(
        description="Stage-1 multi-view pseudo-label generation for YoloSeg."
    )
    parser.add_argument(
        "--data-root",
        type=str, default="../data/CVC-ClinicDB",
        help=(
            "Dataset root directory containing file_list/, Train/, and Test/. "
            "Stage-1 uses file_list/train_label_frames.txt and Train/{JPEGImages,Annotations}."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2.1_hiera_small.pt",
        help="Path to the SAM2 checkpoint file.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="Path to the SAM2 config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Runtime device. Default: cuda",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, existing stage-1 output directories will be removed and recreated.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    run_multi_view_pseudo_label_generation(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        config_path=args.cfg,
        device_str=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()