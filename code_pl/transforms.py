from typing import Tuple

import numpy as np

VIEW_ORIGINAL = "original"
VIEW_ROTATE = "rotate"
VIEW_FLIP = "flip"

SUPPORTED_VIEWS: Tuple[str, ...] = (
    VIEW_ORIGINAL,
    VIEW_ROTATE,
    VIEW_FLIP,
)


def validate_view(view: str) -> None:
    """Validate whether the given view name is supported."""
    if view not in SUPPORTED_VIEWS:
        raise ValueError(
            f"Unsupported view: {view}. "
            f"Expected one of {SUPPORTED_VIEWS}."
        )


def apply_transform(image: np.ndarray, view: str) -> np.ndarray:
    """
    Apply the forward geometric transform for a given view.

    Behavior is intentionally kept identical to the original implementation:
    - original: identity
    - rotate: rotate 90 degrees clockwise
    - flip: flip both vertically and horizontally (equivalent to 180° rotation)

    Args:
        image: Input image array of shape (H, W) or (H, W, C).
        view: One of {"original", "rotate", "flip"}.

    Returns:
        Transformed image array.
    """
    validate_view(view)

    if view == VIEW_ORIGINAL:
        return image
    if view == VIEW_ROTATE:
        return np.rot90(image, k=-1, axes=(0, 1))
    if view == VIEW_FLIP:
        return np.flip(image, axis=(0, 1))

    raise RuntimeError(f"Unexpected view after validation: {view}")


def invert_transform(image: np.ndarray, view: str) -> np.ndarray:
    """
    Invert the geometric transform back to the original view.

    Behavior is intentionally kept identical to the original implementation:
    - original: identity
    - rotate: inverse of clockwise 90° is counter-clockwise 90°
    - flip: self-inverse

    Args:
        image: Input transformed image array.
        view: One of {"original", "rotate", "flip"}.

    Returns:
        Image restored to the original view.
    """
    validate_view(view)

    if view == VIEW_ORIGINAL:
        return image
    if view == VIEW_ROTATE:
        return np.rot90(image, k=1, axes=(0, 1))
    if view == VIEW_FLIP:
        return np.flip(image, axis=(0, 1))

    raise RuntimeError(f"Unexpected view after validation: {view}")