import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = (".png", ".npy")


def _load_grayscale(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".png":
        array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise ValueError(f"Failed to read PNG file: {path}")
    elif extension == ".npy":
        array = np.load(path)
        array = np.asarray(array)
        if array.ndim == 3 and 1 in array.shape:
            array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(
                f"NPY training samples must be 2-D grayscale arrays, got "
                f"shape {array.shape} from: {path}"
            )
    else:
        raise ValueError(f"Unsupported training file type: {path}")

    return array.astype(np.float32, copy=False)


class ImageDataset(Dataset):
    """Paired single-channel training data from input_crop/ and mask_crop/.

    PNG and NPY files use the same preprocessing path after loading:
    float32 -> validity check -> divide by 255 -> [1, H, W] tensor.
    """

    def __init__(self, hp, train=True, transform=None):
        self.path = hp.train if train else hp.valid
        self.image_list = self._read_images()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def _read_images(self):
        input_dir = os.path.join(self.path, "input_crop")
        image_list = []
        for extension in SUPPORTED_EXTENSIONS:
            image_list.extend(glob.glob(os.path.join(input_dir, f"*{extension}")))
        random.shuffle(image_list)
        return image_list

    @staticmethod
    def _mask_path(image_path):
        return image_path.replace(
            f"{os.sep}input_crop{os.sep}",
            f"{os.sep}mask_crop{os.sep}",
        )

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path = self._mask_path(image_path)

        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Matching mask was not found for {image_path}. Expected: {mask_path}"
            )

        image = _load_grayscale(image_path)
        mask = _load_grayscale(mask_path)

        if image.shape != mask.shape:
            raise ValueError(
                f"Input/mask shape mismatch: {image.shape} vs {mask.shape} "
                f"for {image_path}"
            )

        # Preserve the original training-data validity rule exactly.
        if np.max(image) > 1 and np.max(mask) > 1:
            sample = {"sat_img": image, "map_img": mask}
        else:
            sample = {
                "sat_img": np.zeros(image.shape, dtype=np.float32),
                "map_img": np.zeros(mask.shape, dtype=np.float32),
            }

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensorTarget:
    """Apply the original /255 preprocessing and add the channel dimension."""

    def __call__(self, sample):
        sat_img = np.ascontiguousarray(sample["sat_img"] / 255.0, dtype=np.float32)
        map_img = np.ascontiguousarray(sample["map_img"] / 255.0, dtype=np.float32)

        return {
            "sat_img": torch.from_numpy(sat_img).unsqueeze(0),
            "map_img": torch.from_numpy(map_img).unsqueeze(0),
        }
