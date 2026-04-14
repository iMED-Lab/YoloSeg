import os
from typing import List, Optional

import numpy as np
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(filename: str) -> bool:
    """Return True if the filename has a supported image extension."""
    return os.path.splitext(filename)[-1].lower() in IMG_EXTENSIONS


def list_image_files(directory: str) -> List[str]:
    """
    List image filenames in a directory, sorted lexicographically.

    Args:
        directory: Directory containing image files.

    Returns:
        Sorted list of image filenames (not absolute paths).

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    return sorted([name for name in os.listdir(directory) if is_image_file(name)])


def load_image(path: str, mode: str = "RGB") -> np.ndarray:
    """
    Load an image file and convert it to a NumPy array.

    Args:
        path: Image file path.
        mode: PIL convert mode, e.g. "RGB" or "L".

    Returns:
        Image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file does not exist: {path}")

    with Image.open(path) as img:
        return np.array(img.convert(mode))


def save_image(array: np.ndarray, path: str, mode: Optional[str] = None) -> None:
    """
    Save a NumPy array as an image.

    This function preserves the behavior of the original implementation:
    - bool arrays are converted to uint8 with values {0, 255}
    - integer arrays are cast to uint8
    - if `mode` is provided, the PIL image is converted before saving

    Args:
        array: Input NumPy array.
        path: Output image path.
        mode: Optional PIL mode such as "L" or "RGB".
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"`array` must be a numpy.ndarray, got {type(array)}")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.uint8)

    image = Image.fromarray(array)
    if mode is not None:
        image = image.convert(mode)
    image.save(path)