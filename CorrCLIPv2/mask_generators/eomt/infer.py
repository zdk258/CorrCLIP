import sys
import os

# Directory containing this file; used for config paths and to make
# eomt-internal modules (training.*, models.*) importable.
current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)

import yaml
from lightning import seed_everything
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import warnings
import importlib
seed_everything(0, verbose=False)


def get_eomt(cfg_file, use_compile):
    if cfg_file == 'coco_panoptic_eomt_base_640_dinov3':
        config_path = os.path.join(current_dir, "configs/dinov3/coco/panoptic/eomt_base_640_2x.yaml")
    elif cfg_file == 'coco_panoptic_eomt_large_640_dinov3':
        config_path = os.path.join(current_dir, "configs/dinov3/coco/panoptic/eomt_large_640.yaml")
    elif cfg_file == 'coco_panoptic_eomt_small_640_dinov3':
        config_path = os.path.join(current_dir, "configs/dinov3/coco/panoptic/eomt_small_640_2x.yaml")
    elif cfg_file == 'coco_panoptic_eomt_small_640_2x':
        config_path = os.path.join(current_dir, "configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml")
    elif cfg_file == 'coco_panoptic_eomt_base_640_2x':
        config_path = os.path.join(current_dir, "configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml")
    elif cfg_file == 'coco_panoptic_eomt_large_640':
        config_path = os.path.join(current_dir, "configs/dinov2/coco/panoptic/eomt_large_640.yaml")
    elif cfg_file == 'coco_panoptic_eomt_large_1280_dinov3':
        config_path = os.path.join(current_dir, "configs/dinov3/coco/panoptic/eomt_large_1280.yaml")
    elif cfg_file == 'ade20k_panoptic_eomt_large_640':
        config_path = os.path.join(current_dir, "configs/dinov2/ade20k/panoptic/eomt_large_640.yaml")

    if 'ade20k' in cfg_file:
        data_num_classes=150
    elif 'coco' in cfg_file:
        data_num_classes=133
    if '1280' in cfg_file:
        data_img_size = (1280, 1280)
    elif '640' in cfg_file:
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

    # if 'LOCAL_RANK' in os.environ:
    #     device = int(os.environ['LOCAL_RANK'])
    # else:
    #     device = 0

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

    if name is None:
        warnings.warn("No logger name found in the config. Please specify a model name.")
    else:
        try:
            state_dict_path = hf_hub_download(
                repo_id=f"tue-mps/{name}",
                filename="pytorch_model.bin",
            )

            is_dinov3 = "dinov3" in name

            if is_dinov3:
                model_kwargs["ckpt_path"] = state_dict_path
                model_kwargs["delta_weights"] = True

            model = (
                lit_cls(
                    img_size=data_img_size,
                    num_classes=data_num_classes,
                    network=network,
                    **model_kwargs,
                )
                .eval()
            )

            if not is_dinov3:
                state_dict = torch.load(
                    state_dict_path, map_location="cpu", weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)

        except RepositoryNotFoundError:
            warnings.warn(f"Pre-trained model not found for `{name}`. Please load your own checkpoint.")

    if use_compile:
        model = torch.compile(model)

    return model

