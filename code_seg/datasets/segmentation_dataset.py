import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.transforms import (
    JointSegmentationTrainTransform,
    divergence_mask_to_tensor,
    load_grayscale_mask,
    load_rgb_image,
    pil_mask_to_long_tensor,
    preprocess_image,
    resize_mask,
)


@dataclass
class SegmentationDataPaths:
    data_root: str
    file_list_dir: str
    train_dir: str
    val_dir: str
    test_dir: str

    train_image_dir: str
    train_label_dir: str
    train_pseudo_label_dir: str
    train_divergence_dir: str

    val_image_dir: str
    val_label_dir: str

    test_image_dir: str
    test_label_dir: str

    train_all_list: str
    train_label_list: str
    test_all_list: str


def read_filename_list(file_path: str) -> List[str]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File list not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    if len(filenames) == 0:
        raise ValueError(f"Empty file list: {file_path}")

    return filenames


def resolve_segmentation_data_paths(
    data_root: str,
    pseudo_label_dirname: str = "pl_original",
    divergence_dirname: str = "divergence_mask",
) -> SegmentationDataPaths:
    data_root = os.path.abspath(data_root)

    file_list_dir = os.path.join(data_root, "file_list")
    train_dir = os.path.join(data_root, "Train")
    val_dir = os.path.join(data_root, "Val")
    test_dir = os.path.join(data_root, "Test")

    effective_val_dir = val_dir if os.path.isdir(val_dir) else test_dir

    paths = SegmentationDataPaths(
        data_root=data_root,
        file_list_dir=file_list_dir,
        train_dir=train_dir,
        val_dir=effective_val_dir,
        test_dir=test_dir,
        train_image_dir=os.path.join(train_dir, "JPEGImages"),
        train_label_dir=os.path.join(train_dir, "Annotations"),
        train_pseudo_label_dir=os.path.join(train_dir, pseudo_label_dirname),
        train_divergence_dir=os.path.join(train_dir, divergence_dirname),
        val_image_dir=os.path.join(effective_val_dir, "JPEGImages"),
        val_label_dir=os.path.join(effective_val_dir, "Annotations"),
        test_image_dir=os.path.join(test_dir, "JPEGImages"),
        test_label_dir=os.path.join(test_dir, "Annotations"),
        train_all_list=os.path.join(file_list_dir, "train_all_frames.txt"),
        train_label_list=os.path.join(file_list_dir, "train_label_frames.txt"),
        test_all_list=os.path.join(file_list_dir, "test_all_frames.txt"),
    )

    required_directories = [
        paths.file_list_dir,
        paths.train_image_dir,
        paths.train_label_dir,
        paths.train_pseudo_label_dir,
        paths.train_divergence_dir,
        paths.val_image_dir,
        paths.val_label_dir,
        paths.test_image_dir,
        paths.test_label_dir,
    ]
    for directory in required_directories:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Required directory not found: {directory}")

    required_lists = [
        paths.train_all_list,
        paths.train_label_list,
        paths.test_all_list,
    ]
    for file_path in required_lists:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Required file list not found: {file_path}")

    return paths


class YoloSegTrainDataset(Dataset):
    """
    Training dataset:
        - image is resized to network input size
        - pseudo label is resized to network input size
        - divergence mask is resized to network input size
    """

    def __init__(
        self,
        image_dir: str,
        pseudo_label_dir: str,
        divergence_dir: str,
        file_list_path: str,
        image_size: int = 256,
    ) -> None:
        self.image_dir = image_dir
        self.pseudo_label_dir = pseudo_label_dir
        self.divergence_dir = divergence_dir
        self.image_size = image_size
        self.filenames = read_filename_list(file_list_path)
        self.joint_transform = JointSegmentationTrainTransform()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]

        image = load_rgb_image(os.path.join(self.image_dir, filename))
        pseudo_label = load_grayscale_mask(os.path.join(self.pseudo_label_dir, filename))
        divergence_mask = load_grayscale_mask(os.path.join(self.divergence_dir, filename))

        image, pseudo_label, divergence_mask = self.joint_transform(image, pseudo_label, divergence_mask)

        image = preprocess_image(image, self.image_size)
        pseudo_label = resize_mask(pseudo_label, self.image_size)
        divergence_mask = resize_mask(divergence_mask, self.image_size)

        pseudo_label = pil_mask_to_long_tensor(pseudo_label)
        divergence_mask = divergence_mask_to_tensor(divergence_mask)

        return image, pseudo_label, divergence_mask


class YoloSegEvalDataset(Dataset):
    """
    Validation / test dataset:
        - image is resized to network input size
        - label keeps original resolution
        - prediction should be upsampled back to label size before metric computation
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        file_list_path: str,
        image_size: int = 256,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.filenames = read_filename_list(file_list_path)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        filename = self.filenames[index]

        image = load_rgb_image(os.path.join(self.image_dir, filename))
        label = load_grayscale_mask(os.path.join(self.label_dir, filename))

        label_np = np.array(label, dtype=np.int64)
        original_height, original_width = label_np.shape

        image_tensor = preprocess_image(image, self.image_size)
        label_tensor = torch.from_numpy(label_np)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "filename": filename,
            "original_size": (original_height, original_width),
        }