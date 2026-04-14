from torch.utils.data import Dataset

from datasets.segmentation_dataset import read_filename_list
from datasets.transforms import (
    JointSegmentationTrainTransform,
    load_grayscale_mask,
    load_rgb_image,
    pil_mask_to_long_tensor,
    preprocess_image,
    resize_mask,
)


class LabeledReferenceDataset(Dataset):
    """
    Dataset for the labeled reference image(s) used by CPDA.
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
        self.joint_transform = JointSegmentationTrainTransform()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]

        image = load_rgb_image(f"{self.image_dir}/{filename}")
        label = load_grayscale_mask(f"{self.label_dir}/{filename}")

        # Reuse the same interface; the third item is only a placeholder.
        image, label, _ = self.joint_transform(image, label, label)

        image = preprocess_image(image, self.image_size)
        label = resize_mask(label, self.image_size)
        label = pil_mask_to_long_tensor(label)

        return image, label