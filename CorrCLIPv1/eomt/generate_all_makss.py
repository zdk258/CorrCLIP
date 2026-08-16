import os

from torchvision import tv_tensors

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from PIL import Image
import yaml
from lightning import seed_everything
import torch
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import warnings
import importlib
from pathlib import Path
seed_everything(0, verbose=False)
from tqdm import tqdm
import time
import shutil
def city(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    images_dir = ['/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/cityscapes/leftImg8bit/val/frankfurt',
                  '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/cityscapes/leftImg8bit/val/lindau',
                  '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/cityscapes/leftImg8bit/val/munster']

    current_script_path = os.path.abspath(__file__)
    with open(current_script_path, 'r', encoding='utf-8') as file:
        content = file.read()
    with open(os.path.join(target_path, 'config.txt'), 'w', encoding='utf-8') as new_file:
        new_file.write(content)
    print(f"Script saved to {os.path.join(target_path, 'config.txt')}")

    for dir in images_dir:
        lines = os.listdir(dir)
        for base_name in tqdm(lines):
            img_path = os.path.join(dir, base_name)
            if os.path.exists(os.path.join(target_path, base_name[:-3] + 'npz')):
                continue
            instance_mask = generate_mask(img_path)
            np.savez_compressed(os.path.join(target_path, base_name[:-3] + 'npz'), instance_mask=instance_mask)


def ade(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    images_dir = '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/ade/ADEChallengeData2016/images/validation'

    current_script_path = os.path.abspath(__file__)
    with open(current_script_path, 'r', encoding='utf-8') as file:
        content = file.read()
    with open(os.path.join(target_path, 'config.txt'), 'w', encoding='utf-8') as new_file:
        new_file.write(content)
    print(f"Script saved to {os.path.join(target_path, 'config.txt')}")

    lines = os.listdir(images_dir)
    for base_name in tqdm(lines):
        img_path = os.path.join(images_dir, base_name)
        if os.path.exists(os.path.join(target_path, base_name[:-3] + 'npz')):
            continue
        instance_mask = generate_mask(img_path)
        np.savez_compressed(os.path.join(target_path, base_name[:-3] + 'npz'), instance_mask=instance_mask)


def coco(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    images_dir = '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/coco/images/val2017'

    current_script_path = os.path.abspath(__file__)
    with open(current_script_path, 'r', encoding='utf-8') as file:
        content = file.read()
    with open(os.path.join(target_path, 'config.txt'), 'w', encoding='utf-8') as new_file:
        new_file.write(content)
    print(f"Script saved to {os.path.join(target_path, 'config.txt')}")

    lines = os.listdir(images_dir)
    for base_name in tqdm(lines):
        img_path = os.path.join(images_dir, base_name)
        if os.path.exists(os.path.join(target_path, base_name[:-3] + 'npz')):
            continue
        instance_mask = generate_mask(img_path)
        np.savez_compressed(os.path.join(target_path, base_name[:-3] + 'npz'), instance_mask=instance_mask)


def context(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    images_dir = '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/VOCdevkit/VOC2010/JPEGImages'
    txt_file_path = '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt'

    current_script_path = os.path.abspath(__file__)
    with open(current_script_path, 'r', encoding='utf-8') as file:
        content = file.read()
    with open(os.path.join(target_path, 'config.txt'), 'w', encoding='utf-8') as new_file:
        new_file.write(content)
    print(f"Script saved to {os.path.join(target_path, 'config.txt')}")

    with open(txt_file_path, 'r') as file:
        lines = []
        for line in file:
            lines.append(line.strip())
        for base_name in tqdm(lines):
            img_path = os.path.join(images_dir, f"{base_name}.jpg")
            if os.path.exists(os.path.join(target_path, base_name + '.npz')):
                continue
            instance_mask = generate_mask(img_path)
            np.savez_compressed(os.path.join(target_path, base_name + '.npz'), instance_mask=instance_mask)


def voc(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    txt_file_path = '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    images_dir = '/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/VOCdevkit/VOC2012/JPEGImages'

    current_script_path = os.path.abspath(__file__)
    with open(current_script_path, 'r', encoding='utf-8') as file:
        content = file.read()
    with open(os.path.join(target_path, 'config.txt'), 'w', encoding='utf-8') as new_file:
        new_file.write(content)
    print(f"Script saved to {os.path.join(target_path, 'config.txt')}")

    with open(txt_file_path, 'r') as file:
        lines = []
        for line in file:
            lines.append(line.strip())
        for base_name in tqdm(lines):
            img_path = os.path.join(images_dir, f"{base_name}.jpg")
            if os.path.exists(os.path.join(target_path, base_name + '.npz')):
                continue
            instance_mask = generate_mask(img_path)
            np.savez_compressed(os.path.join(target_path, base_name + '.npz'), instance_mask=instance_mask)

def generate_mask(img):
    img = tv_tensors.Image(Image.open(img).convert("RGB"))
    with torch.inference_mode(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]

        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
        mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)

        preds = model.to_per_pixel_preds_panoptic(
            mask_logits,
            class_logits_per_layer[-1],
            model.stuff_classes,
            model.mask_thresh,
            model.overlap_thresh,
        )[0].cpu()

    pred = preds.numpy().astype(np.int16)
    sem_pred, inst_pred = pred[..., 0], pred[..., 1]

    return inst_pred

device = 0  # TODO: change to the GPU you want to use
img_idx = 0  # TODO: change to the index of the image you want to visualize
config_path = "configs/coco/panoptic/eomt_large_640.yaml"  # TODO: change to the config file
data_path = "data/coco"  # TODO: change to the dataset directory
data_num_classes=133
data_img_size = (640, 640)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load encoder
encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
encoder = encoder_cls(img_size=data_img_size, **encoder_cfg.get("init_args", {}))

# Load network
network_cfg = config["model"]["init_args"]["network"]
network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
network_kwargs = {
    k: v for k, v in network_cfg["init_args"].items() if k != "encoder"
}
network = network_cls(
    masked_attn_enabled=False,
    num_classes=data_num_classes,
    encoder=encoder,
    **network_kwargs,
)

# Load Lightning module
lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
model_kwargs = {
    k: v for k, v in config["model"]["init_args"].items() if k != "network"
}
if "stuff_classes" in config["data"].get("init_args", {}):
    model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

model = (
    lit_cls(
        img_size=data_img_size,
        num_classes=data_num_classes,
        network=network,
        **model_kwargs,
    )
    .eval()
    .to(device)
)

name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

if name is None:
    warnings.warn("No logger name found in the config. Please specify a model name.")
else:
    try:
        state_dict_path = hf_hub_download(
            repo_id=f"tue-mps/{name}",
            filename="pytorch_model.bin",
        )
        state_dict = torch.load(
            state_dict_path, map_location=torch.device(f"cuda:{device}"), weights_only=True
        )
        model.load_state_dict(state_dict)
    except RepositoryNotFoundError:
        warnings.warn(f"Pre-trained model not found for `{name}`. Please load your own checkpoint.")

model = torch.compile(model)
model_name = config_path.split('/')[-1].split('.')[0]
target_path = f'/mnt/c/software/Code/code_debug/CorrCLIP/ProxyCLIP-main/data/eomt/{model_name}/'
datasets = ['voc', 'context', 'coco', 'city', 'ade', ]
for dataset in datasets:
    print(f"Checking function: {dataset}")
    func = globals().get(dataset, None)
    if func and callable(func):
        func(target_path + dataset)
        print(f"{dataset} is callable.")
    else:
        print(f"{dataset} is NOT callable.")
