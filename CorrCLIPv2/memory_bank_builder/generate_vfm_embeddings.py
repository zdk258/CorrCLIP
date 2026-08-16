import gc
import os
import argparse
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from detectron2.data.detection_utils import _apply_exif_orientation
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))                             # repo root: vfms, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent))   # this directory: pt_store / manifest

import manifest as mf
from pt_store import PtSlotWriter
from vfms import VFM_REGISTRY, build_vfm

# ============================== Configuration ==============================
parser = argparse.ArgumentParser(description="A script that processes a dataset with specified masks and a pretrained VFM model.")
parser.add_argument('--model', default='radiov3', type=str, choices=list(VFM_REGISTRY.keys()),
                    help='Which VFM to extract features with (drives the retrieval index).')
parser.add_argument('--data_name', default='coco', type=str, help='Bank name (baked into all output filenames).')
parser.add_argument('--img', default='data/coco/images/train2017', type=str, help='The directory containing the images.')
parser.add_argument('--mask_model', default='entityseg', type=str, help='Mask-generation model name; masks are read from memory_bank/masks/{data}/{mask_model}/ by default.')
parser.add_argument('--mask', default=None, type=str, help='Optional: explicit mask directory, overriding the default memory_bank/masks/{data}/{mask_model}/.')
args = parser.parse_args()

# --- CLI arguments ---
model_name = args.model
EMBED_DIM = VFM_REGISTRY[model_name]['dim']
dataset = args.data_name
IMAGE_DIR = os.path.abspath(args.img)
mask_model = args.mask_model  # mask-model tag baked into output filenames
REGION_MASKS_PATH = os.path.abspath(args.mask) if args.mask else None
os.chdir(_ROOT)                                  # outputs are relative to the repo root, so the script can run from any cwd
if REGION_MASKS_PATH is None:
    REGION_MASKS_PATH = os.path.join('memory_bank/masks', dataset, mask_model)

# --- Paths / save directories ---
SAVE_DIR = 'memory_bank/vfm_embeddings'          # raw vectors (source of truth) and the FAISS cache both live here
MANIFESTS_DIR = 'memory_bank/manifests'
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# --- Fixed hyperparameters ---
RANDOM_SEED = 42

# --- FAISS index hyperparameters (m/nbits/nlist are baked into the .faiss file) ---
FAISS_M = 32
FAISS_NBITS = 8
FAISS_NLIST_FACTOR = 8

# --- Output naming (flat: every file sits directly under SAVE_DIR) ---
REGION_SET = dataset                                 # bank name; same for CLIP/VFM so files pair up at a glance
BASE = f'{REGION_SET}_{model_name}'                  # flat filename stem (no extension), e.g. coco_radiov3

# --- Derived at runtime ---
img_num = len(os.listdir(IMAGE_DIR))
SAVE_INTERVAL = img_num // 10 // torch.cuda.device_count()
# =====================================================================


class RegionFeatureExtractor:
    """Dense VFM features (vfms/ factory — same models and preprocessing as deployment inference), aggregated into per-region features via instance masks."""

    def __init__(self, device: str, model_name: str):
        self.device = device
        self.vfm = build_vfm(model_name, device)
        tqdm.write(f"[{self.device}] {model_name} model loaded and moved to {self.device}.")

    def extract_from_paths(self, image_path: str, mask_path: str) -> list:
        """Extract the mean VFM feature of every region given an image path and a mask path.

        Returns:
            list[Tensor]: Region feature vectors (CPU, fp16) ordered by instance idx
                (nonzero torch.unique values, ascending — same order as the manifest);
                the caller stores the idx-th vector at id = start_id + idx.
        """
        # 1. Load data from the given paths
        with np.load(mask_path) as data:
            instance_masks = data['instance_mask']
        instance_masks = torch.from_numpy(instance_masks).to(self.device)

        img = Image.open(image_path).convert("RGB")
        img = _apply_exif_orientation(img)

        # 2. Preprocessing + 3. model inference (per-model branches live in the vfms factory)
        model_feats, img_size = self.vfm.extract_dense(img)

        torch.cuda.empty_cache()

        # 4. Feature post-processing
        try:
            model_feats = F.interpolate(model_feats, size=img_size, mode='bilinear')
            model_feats = F.normalize(model_feats, dim=1)
            model_feats = model_feats.permute(0, 2, 3, 1)  # [1, H, W, C]

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise  # re-raise non-OOM errors instead of leaving a half-built model_feats
            torch.cuda.empty_cache()
            print('GPU out of memory; falling back to CPU...')
            # Move tensors to CPU
            model_feats = model_feats.cpu().float()
            instance_masks = instance_masks.cpu()

            model_feats = F.interpolate(model_feats, size=img_size, mode='bilinear')
            model_feats = F.normalize(model_feats, dim=1)
            model_feats = model_feats.permute(0, 2, 3, 1)  # [1, H, W, C]

        # Mask and feature-map sizes may differ (e.g. pe's ref_size is its post-preprocessing size); align the mask to the feature map
        if instance_masks.shape[-2:] != model_feats.shape[1:3]:
            instance_masks = F.interpolate(
                instance_masks[None, None].float(), size=model_feats.shape[1:3], mode='nearest'
            )[0, 0].to(instance_masks.dtype)

        # 5. Aggregate features per mask (list returned in idx order)
        feats = []

        if len(instance_masks.shape) == 2:
            instance_masks = instance_masks.to(non_blocking=True).int()
            unique_values = torch.unique(instance_masks)
            unique_values = unique_values[unique_values != 0]
            for v in unique_values:
                instance_mask = (instance_masks == v)
                feats.append(model_feats[0][instance_mask].mean(0).half().cpu())
        elif len(instance_masks.shape) == 3:
            for instance_mask in instance_masks:
                feats.append(model_feats[0][instance_mask].mean(0).half().cpu())

        del model_feats, instance_masks
        torch.cuda.empty_cache()

        return feats


import faiss

# --- Main worker (one process per GPU runs this) ---
def process_images_on_gpu(gpu_id, work_subset, config):
    """Process a batch of images on one GPU, writing each region's vector into the shared slot store (memmap) at its manifest id.

    Args:
        gpu_id (int): GPU index.
        work_subset (list): [(full_img_path, stem, start_id, num_regions), ...].
        config (dict): Paths and settings.
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    # Unpack the config
    store = PtSlotWriter(config['store_base'], config['store_n'], config['store_dim'])
    store.open_for_write()
    region_masks_path = config['region_masks_path']
    save_interval = config['save_interval']
    error_log_path = config['error_log_path']
    lock = config['lock']

    try:
        extractor = RegionFeatureExtractor(device=device, model_name=config['model_name'])
    except Exception as e:
        tqdm.write(f"[GPU {gpu_id}] FAILED to initialize model: {e}")
        return

    features_to_save = {}
    # --- counter of images processed since the last save ---
    images_processed_since_last_save = 0
    total_saved = 0
    pbar = tqdm(work_subset, desc=f"GPU {gpu_id} Processing", position=gpu_id, leave=True)

    for full_img_path, stem, start_id, n_expected in pbar:
        try:
            instance_masks_path = os.path.join(region_masks_path, stem + '.npz')
            if not os.path.exists(instance_masks_path):
                continue

            feats = extractor.extract_from_paths(full_img_path, instance_masks_path)
            if len(feats) != n_expected:
                raise ValueError(f"Region count mismatch: extracted {len(feats)} but manifest records {n_expected} (masks may have changed)")
            for idx, vec in enumerate(feats):
                features_to_save[start_id + idx] = vec  # key = stable id (= slot row index)
        except Exception as e:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_type = type(e).__name__
            error_details = str(e).replace('\n', ' ')

            console_msg = (
                f"[{current_time}] [GPU {gpu_id}] Error on image: {stem}. "
                f"Type: {error_type}. Details: {error_details}. Skipping."
            )
            tqdm.write(console_msg)

            log_msg = (
                f"[{current_time}] [GPU {gpu_id}] Image: {stem} | "
                f"Error Type: {error_type} | Details: {error_details}\n"
            )

            with lock:
                with open(error_log_path, 'a') as f:
                    f.write(log_msg)

            torch.cuda.empty_cache()
            gc.collect()
            # Keep going even if this image failed;
            # the counter is still bumped at the end of the loop.

        # --- bump the image counter on every iteration ---
        images_processed_since_last_save += 1

        # --- save condition is based on the image counter ---
        if images_processed_since_last_save >= save_interval and features_to_save:
            n_batch = len(features_to_save)
            ids = list(features_to_save.keys())
            store.write(ids, torch.stack([features_to_save[i] for i in ids]))
            store.flush()
            total_saved += n_batch
            features_to_save.clear()
            # --- reset the image counter ---
            images_processed_since_last_save = 0
            # report progress via the pbar postfix; tqdm.write would add lines and scramble the multi-GPU bars
            pbar.set_postfix_str(f"written {total_saved}(+{n_batch})")

    # Final save after the loop finishes
    if features_to_save:
        n_batch = len(features_to_save)
        ids = list(features_to_save.keys())
        store.write(ids, torch.stack([features_to_save[i] for i in ids]))
        store.flush()
        total_saved += n_batch
        pbar.set_postfix_str(f"written {total_saved}(+{n_batch}) final")

    tqdm.write(f"[GPU {gpu_id}] Processing finished. {total_saved} vectors written to {store.data_path}")


def update_faiss_cache(pt_path, faiss_file_path):
    """Incrementally update the FAISS retrieval cache (IDMap2, add_with_ids) from the raw vector bank ({BASE}.pt, row index = id, source of truth).
    First run: train IVFPQ, build the index, and add all vectors; later runs only add new ids. search() therefore returns manifest ids directly."""
    data = torch.load(pt_path, map_location='cpu')
    all_ids = list(range(data.shape[0]))
    if not all_ids:
        print("Vector bank is empty; skipping FAISS."); return

    if os.path.exists(faiss_file_path):
        index = faiss.read_index(faiss_file_path)
        existing = set(faiss.vector_to_array(index.id_map).astype('int64').tolist())  # ids already present in the IDMap2
    else:
        index, existing = None, set()

    new_ids = [i for i in all_ids if i not in existing]
    if not new_ids:
        print(f"FAISS already contains all {len(all_ids)} vectors; nothing to update."); return

    vecs = data[torch.as_tensor(new_ids, dtype=torch.long)].float().numpy()
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)   # normalize before inner-product search
    vecs = np.ascontiguousarray(vecs, dtype='float32')
    ids64 = np.asarray(new_ids, dtype='int64')
    dim = vecs.shape[1]

    if index is None:
        if dim % FAISS_M != 0:
            raise ValueError(f"m={FAISS_M} does not divide dimension {dim}.")
        nlist = max(1, int(FAISS_NLIST_FACTOR * np.sqrt(len(all_ids))))
        ivf = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, FAISS_M, FAISS_NBITS)
        print(f"Building FAISS from scratch: training nlist={nlist}, m={FAISS_M}, nbits={FAISS_NBITS} ...")
        try:                                  # GPU can speed up training; move back to CPU afterwards, then wrap in IDMap2
            res = faiss.StandardGpuResources()
            gpu_ivf = faiss.index_cpu_to_gpu(res, int(torch.cuda.current_device()), ivf)
            gpu_ivf.train(vecs)
            ivf = faiss.index_gpu_to_cpu(gpu_ivf)
        except Exception as e:
            print(f"GPU training unavailable ({e}); training on CPU instead.")
            ivf.train(vecs)
        index = faiss.IndexIDMap2(ivf)
        index.add_with_ids(vecs, ids64)
    else:
        print(f"Incrementally updating FAISS: add_with_ids with {len(new_ids)} new vectors ...")
        index.add_with_ids(vecs, ids64)

    faiss.write_index(index, faiss_file_path)
    print(f"FAISS cache updated: {faiss_file_path} ({index.ntotal} vectors total).")


# --- Main launcher ---
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    store_base = os.path.join(SAVE_DIR, BASE)                   # raw vector bank (source of truth): {BASE}.pt, row index = id
    faiss_file_path = os.path.join(SAVE_DIR, f'{BASE}.faiss')   # FAISS retrieval cache
    error_log_path = os.path.join(SAVE_DIR, f'{BASE}.error.log')

    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Error: no CUDA device detected.")
        return

    # 1) Load the manifest (run build_manifest first): it fixes each region's stable id
    manifest_txt, images_tsv = mf.paths(MANIFESTS_DIR, REGION_SET)
    if not os.path.exists(images_tsv):
        print(f"Manifest not found: {images_tsv}")
        print(f"Run first: python memory_bank_builder/build_manifest.py --data_name {dataset} --mask_model {mask_model} --img {IMAGE_DIR}")
        sys.exit(1)
    images = mf.load_images(images_tsv)
    total = mf.total_regions(images)
    print(f"manifest: {len(images)} images, {total} regions (vectors).")

    # 2) stem -> image path (recursive scan of IMAGE_DIR)
    stem_to_img = {}
    for root, _, files in os.walk(IMAGE_DIR):
        for name in files:
            if name.lower().endswith(IMG_EXTS):
                stem_to_img.setdefault(Path(name).stem, os.path.join(root, name))

    # 3) Slot store (resume reuses the intermediate memmap; a finished .pt plus a grown manifest gets its old rows imported for incremental extension)
    store = PtSlotWriter(store_base, total, EMBED_DIM)
    store.create()
    done_slots = store.done_mask()

    work, missing = [], 0
    for stem, start_id, n in images:
        if done_slots[start_id:start_id + n].all():
            continue
        if stem not in stem_to_img:
            missing += 1
            continue
        work.append((stem_to_img[stem], stem, start_id, n))
    if missing:
        print(f"Warning: {missing} registered images not found under IMAGE_DIR; skipping them.")

    if not work:
        print("All registered images already processed.")
    else:
        print(f"{len(work)} images to process; found {num_gpus} GPUs, splitting work...")
        random.seed(RANDOM_SEED)
        random.shuffle(work)
        chunks = [work[i::num_gpus] for i in range(num_gpus)]
        lock = mp.Lock()

        config = {
            'store_base': store_base,
            'store_n': total,
            'store_dim': EMBED_DIM,
            'region_masks_path': REGION_MASKS_PATH,
            'save_interval': SAVE_INTERVAL,
            'error_log_path': error_log_path,
            'lock': lock,
            'model_name': model_name,
        }

        processes = []
        for gpu_id in range(num_gpus):
            if chunks[gpu_id]:
                p = mp.Process(target=process_images_on_gpu, args=(gpu_id, chunks[gpu_id], config))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()
        print("\n--- All workers finished; raw vectors written to the slot store by id ---")

    # All slots done -> finalize {BASE}.pt (source of truth), then incrementally update the FAISS cache from it
    if store.finalize():
        update_faiss_cache(store.pt_path, faiss_file_path)
        print(f"Use the manifest to map ids back to images/masks: {manifest_txt}")
    else:
        print("Some slots are unfinished (failed images? see error.log); .pt not finalized / FAISS not updated. Fix and rerun the same command to resume.")
        sys.exit(1)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
