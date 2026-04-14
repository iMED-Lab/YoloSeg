import argparse
import logging
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from datasets.reference_dataset import LabeledReferenceDataset
from datasets.segmentation_dataset import (
    YoloSegEvalDataset,
    resolve_segmentation_data_paths,
    YoloSegTrainDataset,
)
from networks.unet import UNet
from utils.checkpoint import save_model_weights
from utils.displacement import apply_cpda
from utils.losses import (
    MaskedCrossEntropyLoss,
    MaskedSymmetricKLLoss,
    compute_consensus_mask,
)
from utils.metrics import MetricTracker, logits_to_prediction


def set_random_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_rampup(current_epoch: int, start_epoch: int, end_epoch: int) -> float:
    if current_epoch <= start_epoch:
        return 0.0
    if current_epoch >= end_epoch:
        return 1.0
    return float(current_epoch - start_epoch) / float(end_epoch - start_epoch)


def build_reference_dataset_to_match_training_length(
    reference_dataset: LabeledReferenceDataset,
    target_length: int,
):
    if len(reference_dataset) == target_length:
        return reference_dataset

    repeat_factor = target_length // len(reference_dataset)
    repeated_dataset = ConcatDataset([reference_dataset for _ in range(repeat_factor)])

    remaining = target_length - len(repeated_dataset)
    if remaining > 0:
        repeated_dataset = ConcatDataset(
            [repeated_dataset, Subset(reference_dataset, range(remaining))]
        )

    return repeated_dataset


def build_model(in_channels: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = UNet(in_chns=in_channels, class_num=num_classes)
    return model.to(device)


def apply_consistency_transforms(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flip_dims = random.choice([[2], [3], [2, 3]])
    images_flip = torch.flip(images, dims=flip_dims)
    logits_flip = model(images_flip)
    logits_flip = torch.flip(logits_flip, dims=flip_dims)

    rotation_k = random.randint(1, 2)
    images_rot = torch.rot90(images, rotation_k, dims=[2, 3])
    logits_rot = model(images_rot)
    logits_rot = torch.rot90(logits_rot, -rotation_k, dims=[2, 3])

    return logits_flip, logits_rot


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    metric_tracker = MetricTracker()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]
            original_sizes = batch["original_size"]

            logits = model(images.float())
            batch_size = images.shape[0]

            original_heights, original_widths = original_sizes

            for batch_idx in range(batch_size):
                original_height = int(original_heights[batch_idx])
                original_width = int(original_widths[batch_idx])

                resized_logits = F.interpolate(
                    logits[batch_idx: batch_idx + 1],
                    size=(original_height, original_width),
                    mode="bilinear",
                    align_corners=False,
                )

                prediction = logits_to_prediction(resized_logits).squeeze(0).cpu().numpy()
                label = labels[batch_idx].cpu().numpy()

                metric_tracker.update(label, prediction)

    return metric_tracker.compute()


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    reference_loader: DataLoader,
    optimizer: optim.Optimizer,
    ce_loss_fn: MaskedCrossEntropyLoss,
    kl_loss_fn: MaskedSymmetricKLLoss,
    device: torch.device,
    consistency_weight: float,
    rampup_ratio: float,
    base_lr: float,
    total_iters: int,
    iter_num: int,
) -> Tuple[dict, int]:
    model.train()

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    kl_loss_sum = 0.0

    reference_iter = iter(reference_loader)

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, pseudo_labels, divergence_masks in progress_bar:
        images = images.to(device)
        pseudo_labels = pseudo_labels.to(device)
        divergence_masks = divergence_masks.to(device)

        try:
            ref_images, ref_labels = next(reference_iter)
        except StopIteration:
            reference_iter = iter(reference_loader)
            ref_images, ref_labels = next(reference_iter)

        ref_images = ref_images.to(device)
        ref_labels = ref_labels.to(device)

        images, pseudo_labels, divergence_masks = apply_cpda(
            source_images=images,
            source_labels=pseudo_labels,
            source_divergence_masks=divergence_masks,
            reference_images=ref_images,
            reference_labels=ref_labels,
        )

        logits = model(images.float())
        logits_flip, logits_rot = apply_consistency_transforms(model, images.float())

        consensus_mask = compute_consensus_mask(divergence_masks)

        loss_ce = (
            ce_loss_fn(logits, pseudo_labels, consensus_mask)
            + ce_loss_fn(logits_flip, pseudo_labels, consensus_mask)
            + ce_loss_fn(logits_rot, pseudo_labels, consensus_mask)
        ) / 3.0

        probs = torch.softmax(logits, dim=1)
        probs_flip = torch.softmax(logits_flip, dim=1)
        probs_rot = torch.softmax(logits_rot, dim=1)

        loss_kl = (
            kl_loss_fn(probs, probs_flip, divergence_masks)
            + kl_loss_fn(probs, probs_rot, divergence_masks)
            + kl_loss_fn(probs_flip, probs_rot, divergence_masks)
        ) / 3.0

        total_loss = loss_ce + consistency_weight * rampup_ratio * loss_kl

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        current_lr = base_lr * (1.0 - iter_num / total_iters) ** 0.9
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        iter_num += 1

        total_loss_sum += total_loss.item()
        ce_loss_sum += loss_ce.item()
        kl_loss_sum += loss_kl.item()

        progress_bar.set_postfix(
            iter=iter_num,
            lr=f"{current_lr:.6f}",
            total_loss=f"{total_loss.item():.4f}",
            ce=f"{loss_ce.item():.4f}",
            kl=f"{loss_kl.item():.4f}",
            w=f"{consistency_weight * rampup_ratio:.4f}",
        )

    num_iterations = len(train_loader)
    stats = {
        "total_loss": total_loss_sum / num_iterations,
        "ce_loss": ce_loss_sum / num_iterations,
        "kl_loss": kl_loss_sum / num_iterations,
    }
    return stats, iter_num


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YoloSeg segmentation model.")

    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/CVC-ClinicDB",
        help="Dataset root containing file_list/, Train/, Test/, and optionally Val/.",
    )
    parser.add_argument("--exp-name", type=str, default="YoloSeg/CVC-ClinicDB")
    parser.add_argument("--save-root", type=str, default="checkpoints")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--base-lr", type=float, default=0.01)

    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--rampup-start", type=int, default=10)
    parser.add_argument("--rampup-end", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1, help="Whether to use deterministic training.")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    deterministic = bool(args.deterministic)

    device = get_device()
    set_random_seed(args.seed, deterministic)

    save_dir = os.path.join(args.save_root, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_dir, "train.log"),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    data_paths = resolve_segmentation_data_paths(
        data_root=args.data_root,
        pseudo_label_dirname="pl_original",
        divergence_dirname="divergence_mask",
    )

    train_dataset = YoloSegTrainDataset(
        image_dir=data_paths.train_image_dir,
        pseudo_label_dir=data_paths.train_pseudo_label_dir,
        divergence_dir=data_paths.train_divergence_dir,
        file_list_path=data_paths.train_all_list,
        image_size=args.image_size,
    )

    reference_dataset = LabeledReferenceDataset(
        image_dir=data_paths.train_image_dir,
        label_dir=data_paths.train_label_dir,
        file_list_path=data_paths.train_label_list,
        image_size=args.image_size,
    )
    reference_dataset = build_reference_dataset_to_match_training_length(
        reference_dataset, len(train_dataset)
    )

    val_file_list_path = data_paths.test_all_list
    val_candidate = os.path.join(data_paths.file_list_dir, "val_all_frames.txt")
    if os.path.isdir(os.path.join(args.data_root, "Val")) and os.path.isfile(val_candidate):
        val_file_list_path = val_candidate

    val_dataset = YoloSegEvalDataset(
        image_dir=data_paths.val_image_dir,
        label_dir=data_paths.val_label_dir,
        file_list_path=val_file_list_path,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.in_channels, args.num_classes, device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    ce_loss_fn = MaskedCrossEntropyLoss()
    kl_loss_fn = MaskedSymmetricKLLoss()

    best_dice = -1.0
    iter_num = 0
    total_iters = len(train_loader) * args.epochs

    for epoch in range(args.epochs):
        rampup_ratio = linear_rampup(epoch, args.rampup_start, args.rampup_end)

        train_metrics, iter_num = train_one_epoch(
            model=model,
            train_loader=train_loader,
            reference_loader=reference_loader,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            kl_loss_fn=kl_loss_fn,
            device=device,
            consistency_weight=args.consistency_weight,
            rampup_ratio=rampup_ratio,
            base_lr=args.base_lr,
            total_iters=total_iters,
            iter_num=iter_num,
        )

        val_metrics = validate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"lr={current_lr:.6f} "
            f"train_total={train_metrics['total_loss']:.4f} "
            f"train_ce={train_metrics['ce_loss']:.4f} "
            f"train_kl={train_metrics['kl_loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_jaccard={val_metrics['jaccard']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_model_weights(model, os.path.join(save_dir, "best.pth"))
            logging.info(f"Saved best checkpoint to {os.path.join(save_dir, 'best.pth')}")

        save_model_weights(model, os.path.join(save_dir, "last.pth"))

        if args.save_epoch_checkpoints:
            save_model_weights(model, os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))

    logging.info("Training finished.")


if __name__ == "__main__":
    main()