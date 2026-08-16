import os
import sys
import time
import random
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))   # repo root: mask_generators, etc.

from mask_generators import MASK_GENERATOR_CHOICES, build_mask_generator

# ============================== Configuration ==============================
parser = argparse.ArgumentParser(
    description="Generate instance masks using a chosen segmentation model.",
)
parser.add_argument(
    "--model", choices=MASK_GENERATOR_CHOICES, default="entityseg",
    help=f"Segmentation backbone. One of: {', '.join(MASK_GENERATOR_CHOICES)}",
)
parser.add_argument("--data_name", default="coco", type=str,
                    help="Dataset name, used for the mask output subdirectory {save_dir}/{data}/{model}/")
parser.add_argument("--image_dir", default='data/coco/images/train2017', type=str)
parser.add_argument("--save_dir", default="memory_bank/masks", type=str,
                    help="Root directory for masks; actual output goes to {save_dir}/{data}/{model}/, reflecting dataset + segmentation model")
args = parser.parse_args()

# --- CLI arguments ---
MODEL_NAME = args.model
DATASET = args.data_name
IMAGE_DIR = os.path.abspath(args.image_dir)
if args.save_dir != parser.get_default("save_dir"):
    args.save_dir = os.path.abspath(args.save_dir)
os.chdir(_ROOT)   # weights (e.g. Mask2Former_hornet_3x_576d0b.pth) and default outputs are relative to the repo root, so the script can run from any cwd
SAVE_DIR = os.path.join(args.save_dir, DATASET, MODEL_NAME)   # masks land in {save_dir}/{data}/{model}

# --- Fixed hyperparameters ---
RANDOM_SEED = 42
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")   # image extensions picked up by the recursive scan

# Build-time hyperparameters (deployment-time settings live in corrclip_segmentor.set_mask_generator; both share the mask_generators/ implementations).
# EXIF orientation is always applied at build time, keeping masks aligned with the CLIP/VFM feature steps (which read images via detectron2).
BUILD_KWARGS = {
    "entityseg": dict(autocast_dtype=None, conf_thresh=0.5),
    "eomt":      dict(apply_exif=True),
    "sam":       dict(apply_exif=True, pred_iou_thresh=0.7, stability_score_thresh=0.7),
    "sam2":      dict(apply_exif=True, pred_iou_thresh=0.7, stability_score_thresh=0.7,
                      multimask_output=True),
}
# =====================================================================


def process_images_on_gpu(
    image_paths, gpu_id, save_dir, model_name, log_file_path, lock
):
    """Process the assigned image list on a single GPU."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"Worker process {os.getpid()} started, assigned to GPU {gpu_id} (model={model_name}).")

    mask_generator = build_mask_generator(model_name, device, **BUILD_KWARGS[model_name])

    for img_path in tqdm(
        image_paths, position=gpu_id, desc=f"GPU {gpu_id}", leave=True
    ):
        try:
            torch.cuda.empty_cache()
            instance_mask = mask_generator.generate(img_path).cpu().numpy()
            save_path = os.path.join(save_dir, f"{Path(img_path).stem}.npz")
            np.savez_compressed(save_path, instance_mask=instance_mask)
        except Exception as e:
            print(f"General error processing {img_path} on GPU {gpu_id}: {e}")
            with lock:
                with open(log_file_path, "a") as log_file:
                    error_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    log_file.write(f"[{error_time}] GENERAL ERROR on GPU {gpu_id} while processing image '{img_path}'.\n")
                    log_file.write(f"  Exception: {e}\n\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Copy the current script into the save dir
    current_script = os.path.abspath(__file__)
    target_script = os.path.join(SAVE_DIR, os.path.basename(current_script))
    shutil.copy2(current_script, target_script)
    print(f"Copied the current script to: {target_script}")

    # --- Error-log file path ---
    log_file_path = os.path.join(SAVE_DIR, "processing_error_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        print(f"Removed previous log file: {log_file_path}")
    print(f"Errors will be logged to: {log_file_path}")

    lock = mp.Lock()

    # --- 1. Build the image list (recursive scan) ---
    print("Scanning for images to process...")
    all_image_paths = []
    for root, _, files in os.walk(IMAGE_DIR):
        for name in files:
            all_image_paths.append(os.path.join(root, name))

    images_to_process = [
        path
        for path in all_image_paths
        if not os.path.exists(os.path.join(SAVE_DIR, f"{Path(path).stem}.npz"))
        and path.lower().endswith(IMG_EXTS)
    ]

    if not images_to_process:
        print("No new images to process. All done.")
        exit()

    # --- Shuffle with a fixed seed ---
    print(f"Found {len(images_to_process)} new images to process. Shuffling with fixed seed...")
    random.seed(RANDOM_SEED)
    random.shuffle(images_to_process)

    # --- 2. Split work across GPUs ---
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA-enabled GPUs found. Please run on a single-core CPU or check your PyTorch installation.")
        exit()

    print(f"Found {num_gpus} GPUs. Splitting work...")
    image_splits = np.array_split(images_to_process, num_gpus)

    # --- 3. Launch worker processes ---
    processes = []
    start_time = time.time()

    for gpu_id in range(num_gpus):
        if len(image_splits[gpu_id]) > 0:
            p = mp.Process(
                target=process_images_on_gpu,
                args=(
                    image_splits[gpu_id].tolist(),
                    gpu_id,
                    SAVE_DIR,
                    MODEL_NAME,
                    log_file_path,
                    lock,
                ),
            )
            p.start()
            processes.append(p)

    # --- 4. Wait for all workers to finish ---
    for p in processes:
        p.join()

    end_time = time.time()
    print("\n" * num_gpus)
    print(f"All processes finished. Total time: {end_time - start_time:.2f} seconds.")
