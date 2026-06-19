import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_array(path):
    return np.load(path, mmap_mode="r")


def _as_2d(arr):
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, 0]
    raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")


class PatchDaymetDataset(Dataset):
    """
    Lazily samples paired LR/HR patches from cached Daymet .npy files.

    The cached climate variable arrays are expected to be unscaled, with shape
    (time, y, x) or (time, y, x, channels). Scaling happens after cropping.
    """

    def __init__(
        self,
        path_output,
        scaler,
        split="train",
        lr_patch_size=8,
        scale_factor=4,
        patches_per_image=1,
        elevation=False,
        elevation_hr=False,
        random_patches=True,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if lr_patch_size <= 0:
            raise ValueError("lr_patch_size must be positive")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive")

        self.path_output = path_output
        self.scaler = scaler
        self.split = split
        self.lr_patch_size = int(lr_patch_size)
        self.scale_factor = int(scale_factor)
        self.hr_patch_size = self.lr_patch_size * self.scale_factor
        self.patches_per_image = int(patches_per_image)
        self.elevation = elevation
        self.elevation_hr = elevation_hr
        self.random_patches = random_patches

        self.x = _load_array(os.path.join(path_output, f"x_{split}.npy"))
        self.y = _load_array(os.path.join(path_output, f"y_{split}.npy"))

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"LR/HR time dimensions differ: {self.x.shape[0]} vs {self.y.shape[0]}"
            )

        self.lr_shape = self.x.shape[1:3]
        self.hr_shape = self.y.shape[1:3]
        self._validate_patch_geometry()

        self.elev_lr = None
        self.elev_hr = None
        if elevation:
            self.elev_lr = _load_array(os.path.join(path_output, "elev_lr_scaled.npy"))
        if elevation_hr:
            self.elev_hr = _load_array(os.path.join(path_output, "elev_hr_scaled.npy"))

    @property
    def hr_patch_shape(self):
        return (self.hr_patch_size, self.hr_patch_size)

    @property
    def lr_patch_shape(self):
        return (self.lr_patch_size, self.lr_patch_size)

    def _validate_patch_geometry(self):
        lr_h, lr_w = self.lr_shape
        hr_h, hr_w = self.hr_shape
        if self.lr_patch_size > lr_h or self.lr_patch_size > lr_w:
            raise ValueError(
                f"LR patch {self.lr_patch_size} is larger than LR field {self.lr_shape}"
            )
        if self.hr_patch_size > hr_h or self.hr_patch_size > hr_w:
            raise ValueError(
                f"HR patch {self.hr_patch_size} is larger than HR field {self.hr_shape}"
            )

        max_row = min(lr_h - self.lr_patch_size, (hr_h - self.hr_patch_size) // self.scale_factor)
        max_col = min(lr_w - self.lr_patch_size, (hr_w - self.hr_patch_size) // self.scale_factor)
        if max_row < 0 or max_col < 0:
            raise ValueError(
                "Patch geometry is invalid for LR shape "
                f"{self.lr_shape}, HR shape {self.hr_shape}, scale {self.scale_factor}"
            )
        self.max_row = int(max_row)
        self.max_col = int(max_col)

    def __len__(self):
        return self.x.shape[0] * self.patches_per_image

    def _scale_2d(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        scaled = self.scaler.transform(arr.reshape(-1, 1))
        return scaled.reshape(arr.shape).astype(np.float32, copy=False)

    def _crop_origin(self, sample_index, patch_index):
        if self.random_patches:
            row = np.random.randint(0, self.max_row + 1)
            col = np.random.randint(0, self.max_col + 1)
            return row, col

        n_cols = self.max_col + 1
        offset = patch_index % ((self.max_row + 1) * n_cols)
        return offset // n_cols, offset % n_cols

    def _elevation_slice(self, elev, sample_index, row, col, size):
        if elev.ndim == 4:
            arr = elev[min(sample_index, elev.shape[0] - 1), row:row + size, col:col + size, 0]
        elif elev.ndim == 3:
            arr = elev[min(sample_index, elev.shape[0] - 1), row:row + size, col:col + size]
        elif elev.ndim == 2:
            arr = elev[row:row + size, col:col + size]
        else:
            raise ValueError(f"Unexpected elevation shape: {elev.shape}")
        return np.asarray(arr, dtype=np.float32)[None, :, :]

    def __getitem__(self, index):
        sample_index = index // self.patches_per_image
        patch_index = index % self.patches_per_image
        lr_row, lr_col = self._crop_origin(sample_index, patch_index)
        hr_row = lr_row * self.scale_factor
        hr_col = lr_col * self.scale_factor

        lr = _as_2d(self.x[sample_index])
        hr = _as_2d(self.y[sample_index])

        lr_crop = lr[
            lr_row:lr_row + self.lr_patch_size,
            lr_col:lr_col + self.lr_patch_size,
        ]
        hr_crop = hr[
            hr_row:hr_row + self.hr_patch_size,
            hr_col:hr_col + self.hr_patch_size,
        ]

        x = self._scale_2d(lr_crop)[None, :, :]
        y = self._scale_2d(hr_crop)[None, :, :]

        if self.elevation:
            elev = self._elevation_slice(
                self.elev_lr, sample_index, lr_row, lr_col, self.lr_patch_size
            )
            x = np.concatenate([x, elev], axis=0)

        if self.elevation_hr:
            elev_hr = self._elevation_slice(
                self.elev_hr, sample_index, hr_row, hr_col, self.hr_patch_size
            )
            y = np.concatenate([y, elev_hr], axis=0)

        return torch.from_numpy(x), torch.from_numpy(y)
