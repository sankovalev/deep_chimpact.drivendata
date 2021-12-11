# type: ignore
"""Classes with datasets"""

import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def add_random_padding(x1, y1, x2, y2, ratio=0.1):
    height = y2 - y1
    width = x2 - x1
    assert height > 0 and width > 0
    max_pad_left = min(x1, width * ratio)
    max_pad_right = min(1.0 - x2, width * ratio)
    max_pad_top = min(y1, height * ratio)
    max_pad_bottom = min(1.0 - y2, height * ratio)
    new_x1 = x1 + random.uniform(-max_pad_left, max_pad_left)
    new_x2 = x2 + random.uniform(-max_pad_right, max_pad_right)
    new_y1 = y1 + random.uniform(-max_pad_top, max_pad_top)
    new_y2 = y2 + random.uniform(-max_pad_bottom, max_pad_bottom)
    return new_x1, new_y1, new_x2, new_y2


class BaseChimpactDataset(Dataset):

    DEPTH_IMAGES_FOLDER: str
    ORIG_IMAGES_FOLDER: str
    transforms = None
    BLACK_LIST = "/workdir/data/meta/black_list_v4.txt"

    def __init__(self) -> None:
        super().__init__()
        with open(self.BLACK_LIST, "r") as f:
            self.black_list = json.load(f)

    @staticmethod
    def _make_crop_and_bbox(image, row, random_padding=False):
        x1, y1, x2, y2 = row.get('x1'), row.get('y1'), row.get('x2'), row.get('y2')
        x1, y1 = max(0, x1), max(0, y1)
        if random_padding:
            x1, y1, x2, y2 = add_random_padding(x1, y1, x2, y2)
        bbox = [x1, y1, x2, y2]
        height, width = image.shape[:2]
        x1 = int(x1 * width)
        x2 = int(x2 * width)
        y1 = int(y1 * height)
        y2 = int(y2 * height)
        return image[y1:y2, x1:x2], bbox

    def _prepare_data(self, image, row, random_padding):
        if self.transforms is None:
            raise ValueError
        image, bbox = self._make_crop_and_bbox(image, row, random_padding)
        augmented = self.transforms(image=image)
        image_tensor = augmented['image'] / 255.0
        return {
            "tensor": image_tensor,
            "width": torch.as_tensor([bbox[2] - bbox[0]], dtype=torch.float32),
            "height": torch.as_tensor([bbox[3] - bbox[1]], dtype=torch.float32),
        }

    def _filter_invalid_images(self):
        records = []
        for idx in range(len(self.df)):
            row = self._get_row(idx)
            video_id = row.get('video_id')
            image_name = f"{video_id.split('.')[0]}_{row.get('time')}.png"

            if image_name in self.black_list:
                continue

            filename = os.path.join(self.DEPTH_IMAGES_FOLDER, image_name)
            if not os.path.exists(filename):
                continue

            records.append(dict(**row))
        return pd.DataFrame.from_records(records)

    def _read_image(self, filename):
        img_depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img_orig = cv2.imread(
            filename.replace(self.DEPTH_IMAGES_FOLDER, self.ORIG_IMAGES_FOLDER)
        )
        return np.dstack((img_depth, img_orig))

    def _get_row(self, idx):
        return self.df.iloc[idx]

    def __len__(self) -> int:
        return len(self.df)


class TrainAndValidDataset(BaseChimpactDataset):
    """Train and valid datasets for Chimpact challenge"""

    TRAIN_DF_FOLDER = "/workdir/data/meta/train_v3/"
    VALID_DF_FOLDER = "/workdir/data/meta/valid_v3/"
    DEPTH_IMAGES_FOLDER = "/workdir/data/images/train_dpt_large"
    ORIG_IMAGES_FOLDER = "/workdir/data/images/train"

    def __init__(self,
                 fold_idx,
                 transforms,
                 debug: bool,
                 valid: bool = False,
                 **kwargs) -> None:
        """Init train or valid dataset"""
        super().__init__()
        if valid:
            df_path = os.path.join(self.VALID_DF_FOLDER, f"fold{fold_idx}.csv")
        else:
            df_path = os.path.join(self.TRAIN_DF_FOLDER, f"fold{fold_idx}.csv")
        self.df = pd.read_csv(df_path)
        self.df = self._filter_invalid_images()
        self.fold_idx = fold_idx
        self.transforms = transforms
        self.debug = debug
        self.valid = valid

    def _get_target(self, row):
        distance = row.get('distance')
        if self.valid:
            return distance
        return distance + (random.random() / 2) - 0.25

    def __getitem__(self, idx, return_original=False):
        """
        Returns pair of image data and channels
        """
        row = self._get_row(idx)
        video_id = row.get('video_id').split('.')[0]
        time = row.get('time')
        filename = os.path.join(self.DEPTH_IMAGES_FOLDER, f"{video_id}_{time}.png")
        image = self._read_image(filename)
        if return_original:
            return image, row
        data = self._prepare_data(image, row, not self.valid)
        target = torch.as_tensor([self._get_target(row)])
        return data, target

    def __len__(self) -> int:
        if self.debug:
            return 100
        return len(self.df)


class TestDataset(BaseChimpactDataset):
    """Test dataset for Chimpact challenge"""

    DF_FOLDER = None
    TEST_DF = "/workdir/data/meta/test_metadata_v4.csv"
    DEPTH_IMAGES_FOLDER = "/workdir/data/images/test_dpt_large"
    ORIG_IMAGES_FOLDER = "/workdir/data/images/test"

    def __init__(self,
                 transforms,
                 **kwargs) -> None:
        super().__init__()
        if self.DF_FOLDER is not None:
            df_path = os.path.join(self.DF_FOLDER, f"fold{kwargs.get('fold_idx')}.csv")
        else:
            df_path = self.TEST_DF
        self.df = pd.read_csv(df_path)
        self.df = self._filter_invalid_images()
        self.transforms = transforms

    def __getitem__(self, idx, return_original=False):
        """
        Returns pair of image data and name
        """
        row = self._get_row(idx)
        video_id = row.get('video_id')
        time = row.get('time')
        filename = os.path.join(self.DEPTH_IMAGES_FOLDER, f"{video_id.split('.')[0]}_{time}.png")
        image = self._read_image(filename)
        if return_original:
            return image, row
        data = self._prepare_data(image, row, False)
        return data, video_id, time


class ValidForStackingDataset(TestDataset):
    """Make predictions on folds."""

    DF_FOLDER = "/workdir/data/meta/valid_v3/"
    DEPTH_IMAGES_FOLDER = "/workdir/data/images/train_dpt_large"
    ORIG_IMAGES_FOLDER = "/workdir/data/images/train"
