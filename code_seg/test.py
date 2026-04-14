import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.segmentation_dataset import (
    YoloSegEvalDataset,
    resolve_segmentation_data_paths,
)
from networks.unet import UNet
from utils.checkpoint import load_model_weights
from utils.metrics import MetricTracker, logits_to_prediction


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(in_channels: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = UNet(in_chns=in_channels, class_num=num_classes)
    return model.to(device)


def save_prediction(prediction: np.ndarray, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = Image.fromarray(prediction.astype(np.uint8))
    image.save(save_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test YoloSeg segmentation model.")

    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/CVC-ClinicDB",
        help="Dataset root containing file_list/, Train/, Test/, and optionally Val/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/YoloSeg/CVC-ClinicDB/best.pth",
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pred/CVC-ClinicDB",
        help="Directory to save predicted masks.",
    )

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    data_paths = resolve_segmentation_data_paths(
        data_root=args.data_root,
        pseudo_label_dirname="pl_original",
        divergence_dirname="divergence_mask",
    )

    test_dataset = YoloSegEvalDataset(
        image_dir=data_paths.test_image_dir,
        label_dir=data_paths.test_label_dir,
        file_list_path=data_paths.test_all_list,
        image_size=args.image_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.in_channels, args.num_classes, device)
    model = load_model_weights(model, args.checkpoint, map_location=device)
    model.eval()

    metric_tracker = MetricTracker()

    total_process_time = 0.0
    total_frames = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)

        for batch in progress_bar:
            images = batch["image"].to(device)
            labels = batch["label"]
            filenames = batch["filename"]
            original_sizes = batch["original_size"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            logits = model(images.float())

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_process_time += time.perf_counter() - start_time

            batch_size = images.shape[0]
            total_frames += batch_size

            original_heights, original_widths = original_sizes

            for batch_idx in range(batch_size):
                original_height = int(original_heights[batch_idx])
                original_width = int(original_widths[batch_idx])

                resized_logits = F.interpolate(
                    logits[batch_idx : batch_idx + 1],
                    size=(int(original_height), int(original_width)),
                    mode="bilinear",
                    align_corners=False,
                )

                prediction = logits_to_prediction(resized_logits).squeeze(0).cpu().numpy()
                label = labels[batch_idx].cpu().numpy()

                metric_tracker.update(label, prediction)
                save_prediction(prediction, os.path.join(args.output_dir, filenames[batch_idx]))

    metrics = metric_tracker.compute()

    print("TEST FINISHED")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"Jaccard: {metrics['jaccard']:.4f}")
    print(f"Total processing time (s): {total_process_time:.4f}")
    print(f"Total processed frames: {total_frames}")

    if total_process_time > 0:
        print(f"FPS: {total_frames / total_process_time:.4f}")

    if torch.cuda.is_available():
        print(f"Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2 ** 20):.2f}")
        print(f"Allocated memory (MB): {torch.cuda.memory_allocated() / (2 ** 20):.2f}")


if __name__ == "__main__":
    main()