# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 151)}
INSTANCE_MAPPING = {
    0: 7,
    1: 8,
    2: 10,
    3: 12,
    4: 14,
    5: 15,
    6: 18,
    7: 19,
    8: 20,
    9: 22,
    10: 23,
    11: 24,
    12: 27,
    13: 30,
    14: 31,
    15: 32,
    16: 33,
    17: 35,
    18: 36,
    19: 37,
    20: 38,
    21: 39,
    22: 41,
    23: 42,
    24: 43,
    25: 44,
    26: 45,
    27: 47,
    28: 49,
    29: 50,
    30: 53,
    31: 55,
    32: 56,
    33: 57,
    34: 58,
    35: 62,
    36: 64,
    37: 65,
    38: 66,
    39: 67,
    40: 69,
    41: 70,
    42: 71,
    43: 72,
    44: 73,
    45: 74,
    46: 75,
    47: 76,
    48: 78,
    49: 80,
    50: 81,
    51: 82,
    52: 83,
    53: 85,
    54: 86,
    55: 87,
    56: 88,
    57: 89,
    58: 90,
    59: 92,
    60: 93,
    61: 95,
    62: 97,
    63: 98,
    64: 102,
    65: 103,
    66: 104,
    67: 107,
    68: 108,
    69: 110,
    70: 111,
    71: 112,
    72: 115,
    73: 116,
    74: 118,
    75: 119,
    76: 120,
    77: 121,
    78: 123,
    79: 124,
    80: 125,
    81: 126,
    82: 127,
    83: 129,
    84: 130,
    85: 132,
    86: 133,
    87: 134,
    88: 135,
    89: 136,
    90: 137,
    91: 138,
    92: 139,
    93: 142,
    94: 143,
    95: 144,
    96: 146,
    97: 147,
    98: 148,
    99: 149,
}


class ADE20KPanoptic(LightningDataModule):
    def __init__(
        self,
        path,
        stuff_classes: list[int],
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 150,
        color_jitter_enabled=True,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.stuff_classes = stuff_classes

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, target_instance, stuff_classes, **kwargs):
        masks, labels = [], []

        for label_id in target[0].unique():
            cls_id = label_id.item()

            if cls_id not in CLASS_MAPPING:
                continue

            cls_id = CLASS_MAPPING[cls_id]

            if cls_id not in stuff_classes:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls_id)

        for label_id in target_instance[1].unique():
            if label_id == 0:
                continue

            mask = target_instance[1] == label_id
            cls_id = target_instance[0][mask].unique().item() - 1

            masks.append(mask)
            labels.append(INSTANCE_MAPPING[cls_id])

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_instance_zip_path": Path(self.path, "annotations_instance.zip"),
            "target_parser": self.target_parser,
            "stuff_classes": self.stuff_classes,
            "check_empty_targets": self.check_empty_targets,
        }
        self.train_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/training"
            ),
            target_instance_folder_path_in_zip=Path("./annotations_instance/training"),
            transforms=self.transforms,
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/validation"
            ),
            target_instance_folder_path_in_zip=Path(
                "./annotations_instance/validation"
            ),
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        dataset = self.train_dataset

        return DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
