# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset

CLASS_MAPPING = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
    92: 80,
    93: 81,
    95: 82,
    100: 83,
    107: 84,
    109: 85,
    112: 86,
    118: 87,
    119: 88,
    122: 89,
    125: 90,
    128: 91,
    130: 92,
    133: 93,
    138: 94,
    141: 95,
    144: 96,
    145: 97,
    147: 98,
    148: 99,
    149: 100,
    151: 101,
    154: 102,
    155: 103,
    156: 104,
    159: 105,
    161: 106,
    166: 107,
    168: 108,
    171: 109,
    175: 110,
    176: 111,
    177: 112,
    178: 113,
    180: 114,
    181: 115,
    184: 116,
    185: 117,
    186: 118,
    187: 119,
    188: 120,
    189: 121,
    190: 122,
    191: 123,
    192: 124,
    193: 125,
    194: 126,
    195: 127,
    196: 128,
    197: 129,
    198: 130,
    199: 131,
    200: 132,
}


class COCOPanoptic(LightningDataModule):
    def __init__(
        self,
        path,
        stuff_classes: list[int],
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 133,
        color_jitter_enabled=False,
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

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, labels_by_id, is_crowd_by_id, **kwargs):
        target = target[0, :, :] + target[1, :, :] * 256 + target[2, :, :] * 256**2

        masks, labels, is_crowd = [], [], []

        for label_id in target.unique():
            if label_id.item() not in labels_by_id:
                continue

            cls_id = labels_by_id[label_id.item()]
            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target == label_id)
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[label_id.item()])

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./train2017"),
            target_folder_path_in_zip=Path("./panoptic_train2017"),
            annotations_json_path_in_zip=Path("./annotations/panoptic_train2017.json"),
            target_zip_path_in_zip=Path("./annotations/panoptic_train2017.zip"),
            target_zip_path=Path(self.path, "panoptic_annotations_trainval2017.zip"),
            zip_path=Path(self.path, "train2017.zip"),
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./val2017"),
            target_folder_path_in_zip=Path("./panoptic_val2017"),
            annotations_json_path_in_zip=Path("./annotations/panoptic_val2017.json"),
            target_zip_path_in_zip=Path("./annotations/panoptic_val2017.zip"),
            target_zip_path=Path(self.path, "panoptic_annotations_trainval2017.zip"),
            zip_path=Path(self.path, "val2017.zip"),
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
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
