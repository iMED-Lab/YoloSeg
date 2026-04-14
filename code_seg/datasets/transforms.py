import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JointSegmentationTrainTransform:
    """
    Joint transform for:
        - RGB image
        - semantic label mask
        - divergence mask

    Design:
        - photometric transforms are applied to the image only
        - geometric transforms are applied consistently to all inputs
    """

    def __init__(self) -> None:
        pass

    def _apply_photometric(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.8:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))

        return image

    def _apply_geometric(
        self,
        image: Image.Image,
        label: Image.Image,
        divergence_mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
            divergence_mask = TF.hflip(divergence_mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)
            divergence_mask = TF.vflip(divergence_mask)

        if random.random() < 0.5:
            angle = random.uniform(-180.0, 180.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            label = TF.rotate(label, angle, interpolation=InterpolationMode.NEAREST)
            divergence_mask = TF.rotate(divergence_mask, angle, interpolation=InterpolationMode.NEAREST)

        return image, label, divergence_mask

    def __call__(
        self,
        image: Image.Image,
        label: Image.Image,
        divergence_mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        image = self._apply_photometric(image)
        image, label, divergence_mask = self._apply_geometric(image, label, divergence_mask)
        return image, label, divergence_mask


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    image = TF.to_tensor(image)
    image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
    return image


def resize_mask(mask: Image.Image, image_size: int) -> Image.Image:
    return TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)


def pil_mask_to_long_tensor(mask: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def divergence_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """
    Convert a divergence mask image from {0, 255} to {0, 1}.
    """
    array = np.array(mask, dtype=np.uint8)
    array = (array / 255).astype(np.uint8)
    return torch.from_numpy(array)


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_grayscale_mask(path: str) -> Image.Image:
    return Image.open(path).convert("L")