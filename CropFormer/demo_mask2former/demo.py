# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm

try:
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.logger import setup_logger
    from detectron2.data import MetadataCatalog
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.utils.video_visualizer import VideoVisualizer
    from detectron2.utils.visualizer import ColorMode, Visualizer
except ImportError:
    from detectron2.detectron2.config import get_cfg
    from detectron2.detectron2.data.detection_utils import read_image
    from detectron2.detectron2.projects.deeplab import add_deeplab_config
    from detectron2.detectron2.utils.logger import setup_logger
    from detectron2.detectron2.data import MetadataCatalog
    from detectron2.detectron2.engine.defaults import DefaultPredictor
    from detectron2.detectron2.utils.video_visualizer import VideoVisualizer
    from detectron2.detectron2.utils.visualizer import ColorMode, Visualizer

from ..mask2former import add_maskformer2_config


from .predictor import VisualizationDemo
import torch


# constants
WINDOW_NAME = "mask2former demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


model_list = {"mask2former_swin_tiny_3x": "Mask2Former_swin_tiny_3x_3cacfb.pth",
              "mask2former_swin_large_3x": "Mask2Former_swin_large_w7_3x_dd4543.pth",
              "mask2former_hornet_3x": "Mask2Former_hornet_3x_576d0b.pth"}
model = 'mask2former_hornet_3x'
model_path = model_list[model]

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default=f"configs/entityv2/entity_segmentation/{model}.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
        default=['pikachu.jpg']
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", model_path],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_entityseg(cfg_file, ckpt_path):
    args = argparse.Namespace(
        config_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"configs/entityv2/entity_segmentation/{cfg_file}"
        ),
        opts=["MODEL.WEIGHTS", ckpt_path]
    )
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    return predictor


