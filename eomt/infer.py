import os
import sys

import yaml
from lightning import seed_everything
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import warnings
import importlib
seed_everything(0, verbose=False)


def get_eomt(cfg_file, use_compile):
    config_path = f"eomt/configs/coco/panoptic/{cfg_file}"
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

    if 'LOCAL_RANK' in os.environ:
        device = int(os.environ['LOCAL_RANK'])
    else:
        device = 0

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

    if use_compile:
        model = torch.compile(model)

    return model

