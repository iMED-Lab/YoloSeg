import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

from tqdm import tqdm

from image_io import list_image_files, load_image, save_image
from transforms import (
    VIEW_FLIP,
    VIEW_ORIGINAL,
    VIEW_ROTATE,
    SUPPORTED_VIEWS,
    apply_transform,
    validate_view,
)


@dataclass
class Stage1Paths:
    """
    Container for all stage-1 dataset paths.

    Expected dataset structure:
        data_root/
          file_list/
            train_label_frames.txt
          Train/
            JPEGImages/
            Annotations/
    """

    data_root: str
    file_list_dir: str
    train_dir: str
    frames_dir: str
    masks_dir: str
    label_list_path: str
    output_original_dir: str
    output_rotate_dir: str
    output_flip_dir: str
    output_divergence_dir: str


def resolve_stage1_paths(
    data_root: str,
    split_dir_name: str = "file_list",
    train_dir_name: str = "Train",
    frames_dir_name: str = "JPEGImages",
    masks_dir_name: str = "Annotations",
    label_list_name: str = "train_label_frames.txt",
) -> Stage1Paths:
    """
    Resolve all input/output paths required by stage-1 pseudo-label generation.

    Args:
        data_root: Dataset root directory.
        split_dir_name: Directory containing split txt files.
        train_dir_name: Training directory name.
        frames_dir_name: Subdirectory name for training images.
        masks_dir_name: Subdirectory name for training masks.
        label_list_name: Filename storing the one-shot labeled frame.

    Returns:
        Stage1Paths object containing all relevant paths.

    Raises:
        FileNotFoundError: If required directories/files are missing.
    """
    data_root = os.path.abspath(data_root)
    file_list_dir = os.path.join(data_root, split_dir_name)
    train_dir = os.path.join(data_root, train_dir_name)
    frames_dir = os.path.join(train_dir, frames_dir_name)
    masks_dir = os.path.join(train_dir, masks_dir_name)
    label_list_path = os.path.join(file_list_dir, label_list_name)

    for path in [data_root, file_list_dir, train_dir, frames_dir, masks_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path}")

    if not os.path.isfile(label_list_path):
        raise FileNotFoundError(f"Label list file does not exist: {label_list_path}")

    return Stage1Paths(
        data_root=data_root,
        file_list_dir=file_list_dir,
        train_dir=train_dir,
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        label_list_path=label_list_path,
        output_original_dir=os.path.join(train_dir, "pl_original"),
        output_rotate_dir=os.path.join(train_dir, "pl_rotate"),
        output_flip_dir=os.path.join(train_dir, "pl_flip"),
        output_divergence_dir=os.path.join(train_dir, "divergence_mask"),
    )


def read_single_prompt_filename(label_list_path: str) -> str:
    """
    Read the one-shot prompt filename from train_label_frames.txt.

    Current stage-1 implementation only supports one-shot prompting, so this
    file must contain exactly one non-empty line.

    Args:
        label_list_path: Path to train_label_frames.txt.

    Returns:
        Prompt filename string.

    Raises:
        ValueError: If the file is empty or contains more than one entry.
    """
    with open(label_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) == 0:
        raise ValueError(
            f"No valid entries found in label list file: {label_list_path}"
        )
    if len(lines) > 1:
        raise ValueError(
            "Stage-1 pseudo-label generation currently supports one-shot "
            f"prompting only, but got {len(lines)} entries in: {label_list_path}"
        )

    return lines[0]


def get_prompt_paths(paths: Stage1Paths, prompt_filename: str) -> Tuple[str, str]:
    """
    Resolve the prompt frame path and prompt mask path from a shared filename.

    Args:
        paths: Stage1Paths object.
        prompt_filename: Filename read from train_label_frames.txt.

    Returns:
        Tuple of (prompt_frame_path, prompt_mask_path).

    Raises:
        FileNotFoundError: If either the frame or the mask is missing.
    """
    prompt_frame_path = os.path.join(paths.frames_dir, prompt_filename)
    prompt_mask_path = os.path.join(paths.masks_dir, prompt_filename)

    if not os.path.isfile(prompt_frame_path):
        raise FileNotFoundError(f"Prompt frame not found: {prompt_frame_path}")
    if not os.path.isfile(prompt_mask_path):
        raise FileNotFoundError(f"Prompt mask not found: {prompt_mask_path}")

    return prompt_frame_path, prompt_mask_path


def ensure_output_dirs(paths: Stage1Paths, overwrite: bool = False) -> None:
    """
    Ensure the standard stage-1 output directories exist.

    Output directories:
        Train/pl_original
        Train/pl_rotate
        Train/pl_flip
        Train/divergence_mask

    Args:
        paths: Stage1Paths object.
        overwrite: If True, existing directories are removed and recreated.
    """
    output_dirs = [
        paths.output_original_dir,
        paths.output_rotate_dir,
        paths.output_flip_dir,
        paths.output_divergence_dir,
    ]

    for directory in output_dirs:
        if overwrite and os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)


def get_output_dir_for_view(paths: Stage1Paths, view: str) -> str:
    """
    Return the standard output directory for a given view.

    Args:
        paths: Stage1Paths object.
        view: One of {"original", "rotate", "flip"}.

    Returns:
        Output directory path for that view.
    """
    validate_view(view)

    if view == VIEW_ORIGINAL:
        return paths.output_original_dir
    if view == VIEW_ROTATE:
        return paths.output_rotate_dir
    if view == VIEW_FLIP:
        return paths.output_flip_dir

    raise RuntimeError(f"Unexpected view after validation: {view}")


class TemporaryViewDataset:
    """
    Context manager that prepares a temporary frame directory for a given view,
    and a temporary transformed prompt mask if needed.

    Design notes:
    - For the original view, we reuse the original frame directory and prompt mask.
    - For transformed views, we generate a temporary frame directory containing
      transformed copies of all frames, because the SAM2 video predictor expects
      a frame directory input.
    - We do NOT require users to manually prepare a dedicated prompt-mask folder.
      Instead, we transform only the selected prompt mask on demand.

    Returned values:
        (frame_dir_for_this_view, prompt_mask_path_for_this_view)
    """

    def __init__(
        self,
        frames_dir: str,
        prompt_mask_path: str,
        view: str,
    ):
        validate_view(view)

        self.frames_dir = frames_dir
        self.prompt_mask_path = prompt_mask_path
        self.view = view

        self._temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
        self.temp_root: Optional[str] = None
        self.temp_frame_dir: Optional[str] = None
        self.temp_prompt_mask_path: Optional[str] = None

    def _prepare_transformed_frames(self) -> str:
        assert self.temp_root is not None

        temp_frame_dir = os.path.join(self.temp_root, "frames")
        os.makedirs(temp_frame_dir, exist_ok=True)

        frame_names = list_image_files(self.frames_dir)

        for frame_name in tqdm(
            frame_names,
            desc=f"Preparing {self.view} view frames",
            leave=False,
        ):
            src_path = os.path.join(self.frames_dir, frame_name)
            dst_path = os.path.join(temp_frame_dir, frame_name)

            frame_array = load_image(src_path, mode="RGB")
            transformed_frame = apply_transform(frame_array, self.view)
            save_image(transformed_frame, dst_path, mode="RGB")

        return temp_frame_dir

    def _prepare_transformed_prompt_mask(self) -> str:
        assert self.temp_root is not None

        prompt_mask_name = os.path.basename(self.prompt_mask_path)
        temp_prompt_mask_path = os.path.join(self.temp_root, prompt_mask_name)

        mask_array = load_image(self.prompt_mask_path, mode="L")
        transformed_mask = apply_transform(mask_array, self.view)
        save_image(transformed_mask, temp_prompt_mask_path, mode="L")

        return temp_prompt_mask_path

    def __enter__(self) -> Tuple[str, str]:
        if self.view == VIEW_ORIGINAL:
            return self.frames_dir, self.prompt_mask_path

        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"sam2_view_{self.view}_")
        self.temp_root = self._temp_dir_obj.name

        self.temp_frame_dir = self._prepare_transformed_frames()
        self.temp_prompt_mask_path = self._prepare_transformed_prompt_mask()

        return self.temp_frame_dir, self.temp_prompt_mask_path

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._temp_dir_obj is not None:
            self._temp_dir_obj.cleanup()
            self._temp_dir_obj = None
            self.temp_root = None
            self.temp_frame_dir = None
            self.temp_prompt_mask_path = None