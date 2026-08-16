import gc
import os
import argparse
import random
from datetime import datetime
from pathlib import Path

import mmcv
import mmengine.fileio as fileio
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from mmcv.transforms import to_tensor
from torchvision import transforms
from tqdm import tqdm

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))                             # repo root: open_clip, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent))   # this directory: pt_store / manifest

import open_clip.tokenizer as tokenizer
from myutils import UnNormalize
from open_clip import create_model
from PIL import Image

import manifest as mf
from pt_store import PtSlotWriter

from detectron2.data.detection_utils import read_image, _apply_exif_orientation


# ============================== Configuration ==============================
parser = argparse.ArgumentParser(description="A script that processes a dataset with specified masks and a pretrained CLIP model.")
parser.add_argument('--data_name', default='coco', type=str, help='Bank name (baked into all output filenames).')
parser.add_argument('--img', default='data/coco/images/train2017', type=str, help='The directory containing the images.')
parser.add_argument('--mask_model', default='entityseg', type=str, help='Mask-generation model name; masks are read from memory_bank/masks/{data}/{mask_model}/ by default, and the name goes into the embedding filename.')
parser.add_argument('--mask', default=None, type=str, help='Optional: explicit mask directory, overriding the default memory_bank/masks/{data}/{mask_model}/.')
parser.add_argument('--clip', default='dfn_b', type=str, help='The type of CLIP.')
parser.add_argument('--slide_short', default=448, type=int)
parser.add_argument('--slide_crop', default=336, type=int)
args = parser.parse_args()

# --- CLI arguments ---
dataset = args.data_name
IMAGE_DIR = os.path.abspath(args.img)
mask_model = args.mask_model  # mask-model tag baked into output filenames
REGION_MASKS_PATH = os.path.abspath(args.mask) if args.mask else None
os.chdir(_ROOT)                                  # outputs are relative to the repo root, so the script can run from any cwd
if REGION_MASKS_PATH is None:
    REGION_MASKS_PATH = os.path.join('memory_bank/masks', dataset, mask_model)
clip_pretrained = args.clip
slide_short = args.slide_short
slide_crop = args.slide_crop

# --- Paths / save directories ---
SAVE_DIR = 'memory_bank/clip_embeddings'
MANIFESTS_DIR = 'memory_bank/manifests'
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# --- Fixed hyperparameters ---
RANDOM_SEED = 42
slide_stride = 112
scale = (2048, slide_short)

# --- CLIP model settings derived from --clip ---
if clip_pretrained == 'meta_b':
    clip_name = 'metaclip_b'
    clip_type = 'metaclip_fullcc'
    model_type = 'ViT-B-16-quickgelu'
    EMBED_DIM = 512
elif clip_pretrained == 'meta_l':
    clip_name = 'metaclip_l'
    clip_type = 'metaclip_fullcc'
    model_type = 'ViT-L-14-quickgelu'
    EMBED_DIM = 768
elif clip_pretrained == 'dfn_b':
    clip_name = 'dfnclip_b'
    clip_type = None
    model_type = 'hf-hub:apple/DFN2B-CLIP-ViT-B-16'
    EMBED_DIM = 512
elif clip_pretrained == 'dfn_l':
    clip_name = 'dfnclip_l'
    clip_type = None
    model_type = 'hf-hub:apple/DFN2B-CLIP-ViT-L-14'
    EMBED_DIM = 768

# --- Output naming (flat: every file sits directly under SAVE_DIR) ---
REGION_SET = dataset                                   # bank name; same for CLIP/VFM so files pair up at a glance
BASE = f'{REGION_SET}_{clip_name}'                     # flat filename stem (no extension), e.g. coco_metaclip_b

# --- Derived at runtime ---
img_num = len(os.listdir(IMAGE_DIR))
SAVE_INTERVAL = img_num // 10 // torch.cuda.device_count()
# =====================================================================


class FeatureExtractor:
    """
    Wraps model loading (CLIP + DINO), image reading, preprocessing, and feature extraction.
    """

    def __init__(self, device: str):
        """
        Initialize the feature extractor.

        Args:
            device (str): Device to run the models on, e.g. 'cuda:0'.
        """
        self.device = device

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(self.device)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer.tokenize
        print("CLIP model loaded.")

        # --- 2. Initialize the DINO model ---
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        self.dino.eval().to(self.device)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino = self.dino.half()
        print("DINO model loaded.")

        # --- 3. Register a hook to capture DINO qkv features ---
        self.feat_out = {}
        def _hook_fn_forward_qkv(module, input, output):
            self.feat_out["qkv"] = output
        self.dino._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(_hook_fn_forward_qkv)

        # --- 4. Initialize image transforms ---
        self.data_preprocessor_mean = torch.tensor([122.771, 116.746, 104.094]).view(3, 1, 1).to(self.device)
        self.data_preprocessor_std = torch.tensor([68.501, 66.632, 70.323]).view(3, 1, 1).to(self.device)
        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.scale = scale

        tqdm.write(f"[{self.device}] CLIP model loaded and moved to {self.device}.")

    @torch.inference_mode()
    def forward_feature(self, img, masks):
        if type(img) == list:
            img = img[0]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.half()

        # Forward pass in the model
        self.feat_out.clear()
        feat = self.dino.get_intermediate_layers(imgs_norm, n=1)[-1]

        patch_size = self.dino.patch_embed.patch_size
        feat_shape = (imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size)
        nb_im = feat.shape[0]  # Batch size
        nb_tokens = feat.shape[1]  # Number of tokens

        qkv = self.feat_out["qkv"].reshape(nb_im, nb_tokens, 3, -1).permute(2, 0, 1, 3)
        dino_feats = qkv[0] + qkv[1]  # B, L, C
        dino_feats = dino_feats[:, 1:, ]
        dino_feats = F.normalize(dino_feats, dim=-1)

        image_features = self.clip.encode_image(img.half(), dino_feats=dino_feats, feat_shape=feat_shape, instance_masks=masks)

        image_features = image_features.permute(0, 2, 1).reshape(-1, image_features.shape[-1], *feat_shape)
        image_features = nn.functional.interpolate(image_features, size=img.shape[-2:], mode='bilinear')

        return image_features

    @torch.inference_mode()
    def extract_from_paths(self, image_path: str, mask_path: str) -> list:
        """Extract the mean CLIP feature of every region given an image path and a mask path.

        Returns:
            list[Tensor]: Region feature vectors (CPU, fp16) ordered by instance idx
                (nonzero torch.unique values, ascending — same order as the manifest);
                the caller stores the idx-th vector at id = start_id + idx.
        """
        # 1. Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = _apply_exif_orientation(img)
        img = np.array(img)
        original_shape = img.shape[:2]
        img = mmcv.imrescale(
            img,
            scale=self.scale,
            interpolation='bilinear',
            return_scale=False,
            backend='cv2')
        img = img.transpose(2, 0, 1)
        img = to_tensor(img).contiguous().to(self.device)
        img = img.float()
        img = (img - self.data_preprocessor_mean) / self.data_preprocessor_std
        img = img.unsqueeze(0)

        # 2. Load instance masks
        mask_file = os.path.join(mask_path)
        instance_masks = np.load(mask_file)['instance_mask']

        mask_type = None
        if len(instance_masks.shape) == 2:
            mask_type = 'int'
            instance_masks = instance_masks.astype(int)
            instance_masks = torch.from_numpy(instance_masks).unsqueeze(0).to(self.device)
            instance_masks = instance_masks.int()
            instance_masks_resized = F.interpolate(instance_masks.unsqueeze(1).float(), size=img.shape[2:], mode='nearest').int()
        elif len(instance_masks.shape) == 3:
            mask_type = 'bool'
            instance_mask_bool = torch.from_numpy(instance_masks).to(self.device)
            instance_masks = torch.zeros(instance_mask_bool.shape[1:], dtype=torch.int, device=self.device)
            instance_id = 1
            for m in instance_mask_bool:
                instance_masks[m] = instance_id
                instance_id += 1
            instance_masks_resized = F.interpolate(instance_masks.unsqueeze(0).unsqueeze(1).float(), size=img.shape[2:], mode='nearest').int()

        # 3. slide infer
        h_stride, w_stride = self.slide_stride, self.slide_stride
        h_crop, w_crop = self.slide_crop, self.slide_crop
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        clip_feats = torch.zeros((batch_size, self.clip.visual.output_dim, h_img, w_img), dtype=torch.float16, device=self.device)
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_instance_masks = instance_masks_resized[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                    crop_instance_masks = nn.functional.pad(crop_instance_masks, pad, value=10000)
                crop_clip_feats = self.forward_feature(crop_img, crop_instance_masks)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_clip_feats = crop_clip_feats[:, :, t:t + H, l:l + W]

                clip_feats += nn.functional.pad(crop_clip_feats,
                                                (int(x1), int(clip_feats.shape[3] - x2), int(y1),
                                                 int(clip_feats.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        feats = []

        torch.cuda.empty_cache()

        clip_feats = clip_feats / count_mat

        try:
            clip_feats = nn.functional.interpolate(clip_feats.contiguous(), size=original_shape, mode='bilinear')

            clip_feats = clip_feats.half()
            clip_feats = clip_feats.permute(0, 2, 3, 1)  # B, H, W, C
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise  # re-raise non-OOM errors instead of leaving a half-built clip_feats
            torch.cuda.empty_cache()
            print('GPU out of memory; falling back to CPU...')
            clip_feats = clip_feats.cpu().float()
            instance_masks = instance_masks.cpu()
            clip_feats = nn.functional.interpolate(clip_feats.contiguous(), size=original_shape, mode='bilinear')
            clip_feats = clip_feats.permute(0, 2, 3, 1)  # B, H, W, C

        if mask_type == 'int':
            instance_masks = instance_masks[0]
            unique_values = torch.unique(instance_masks)
            unique_values = unique_values[unique_values!=0]
            for v in unique_values:
                instance_mask = instance_masks == v
                feats.append(F.normalize(clip_feats[0][instance_mask].mean(0), dim=-1).half().cpu())
        elif mask_type == 'bool':
            for instance_mask in instance_mask_bool:
                instance_mask = instance_mask.to(clip_feats.device)  # after the CPU fallback clip_feats may be on CPU; the mask must follow
                feats.append(F.normalize(clip_feats[0][instance_mask].mean(0), dim=-1).half().cpu())

        del clip_feats, instance_masks

        return feats


    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l
        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t
        return l, r, t, b

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
        extractor = FeatureExtractor(device=device)
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


# --- Main launcher ---
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    store_base = os.path.join(SAVE_DIR, BASE)                   # final vector bank: {BASE}.pt, row index = id
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
        print("All registered images already processed; nothing to do.")
    else:
        print(f"{len(work)} images to process; found {num_gpus} GPUs, splitting work...")
        random.seed(RANDOM_SEED)
        random.shuffle(work)
        chunks = [work[i::num_gpus] for i in range(num_gpus)]   # round-robin split; each chunk is a list of (path, stem, start, n)
        lock = mp.Lock()

        config = {
            'store_base': store_base,
            'store_n': total,
            'store_dim': EMBED_DIM,
            'region_masks_path': REGION_MASKS_PATH,
            'save_interval': SAVE_INTERVAL,
            'error_log_path': error_log_path,
            'lock': lock,
        }

        processes = []
        for gpu_id in range(num_gpus):
            if chunks[gpu_id]:
                p = mp.Process(target=process_images_on_gpu, args=(gpu_id, chunks[gpu_id], config))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()
        print("\n--- All workers finished; vectors written to the slot store by id ---")

    # All slots done -> finalize the single {BASE}.pt (read directly by the eval-side PTTensorDataset)
    if store.finalize():
        print(f"Done: {store.pt_path} with {total} vectors. Use the manifest to map ids back to images/masks: {manifest_txt}")
    else:
        print("Some slots are unfinished (failed images? see error.log); .pt not finalized. Fix and rerun the same command to resume.")
        sys.exit(1)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()