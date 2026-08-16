"""
Build or extend the id registry (manifest) of a region set (REGION_SET).

Run after generate_mask and before generate_clip/radio_embeddings; rerun after adding images to extend incrementally.
Naming follows the embedding scripts: masks are read from memory_bank/masks/{data}/{mask_model}/ by default.

Example:
  python memory_bank_builder/build_manifest.py --data_name coco --mask_model entityseg \
      --img data/coco/images/train2017
"""
import os
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # this directory: manifest

import manifest as M

parser = argparse.ArgumentParser(description="Build or extend a region-set manifest (append-only id registry).")
parser.add_argument('--data_name', required=True, type=str, help='Bank name')
parser.add_argument('--img', required=True, type=str, help='Image directory (recursive)')
parser.add_argument('--mask_model', default='entityseg', type=str, help='Mask-generation model; masks default to memory_bank/masks/{data}/{mask_model}/')
parser.add_argument('--mask', default=None, type=str, help='Optional: explicit mask directory, overrides the default')
parser.add_argument('--manifests_dir', default='memory_bank/manifests', type=str)
parser.add_argument('--workers', default=None, type=int,
                    help='Parallel workers for counting regions in mask files; defaults to min(32, CPU cores), 1 = serial')
args = parser.parse_args()

args.img = os.path.abspath(args.img)
if args.mask:
    args.mask = os.path.abspath(args.mask)
os.chdir(_ROOT)   # default mask/manifest paths are relative to the repo root, so the script can run from any cwd
mask_dir = args.mask if args.mask else os.path.join('memory_bank/masks', args.data_name, args.mask_model)
region_set = args.data_name

manifest_txt, images_tsv = M.paths(args.manifests_dir, region_set)
before = len(M.load_region_keys(manifest_txt))

print(f"REGION_SET = {region_set}")
print(f"Image dir  = {args.img}")
print(f"Mask dir   = {mask_dir}")
region_keys, images = M.build_or_extend(args.manifests_dir, region_set, args.img, mask_dir,
                                        workers=args.workers)

print(f"\nmanifest: {manifest_txt}")
print(f"images  : {images_tsv}")
print(f"Images = {len(images)}, total regions (vectors) = {len(region_keys)} ({len(region_keys) - before} added this run)")
