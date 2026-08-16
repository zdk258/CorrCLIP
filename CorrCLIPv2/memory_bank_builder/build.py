"""One-command CorrCLIPv2 memory-bank builder: point it at an image directory to get eval-ready .pt feature banks + a FAISS index.

    python memory_bank_builder/build.py --img /path/to/images --data_name mybank

Pipeline (stops at the first failed step; rerunning the same command resumes where it left off, and rerunning after adding images extends the bank incrementally):
  1. generate_mask            instance masks             -> memory_bank/masks/{data}/{mask_model}/
  2. build_manifest           region id registry         -> memory_bank/manifests/{data}.*
  3. generate_clip_embeddings CLIP region features (xN)  -> memory_bank/clip_embeddings/{data}_{clip}.pt
  4. generate_vfm_embeddings  VFM region features+index  -> memory_bank/vfm_embeddings/{data}_{vfm}.pt / .faiss

Mask models (--mask_model): entityseg / eomt / sam / sam2 (implementations in mask_generators/)
Index VFMs  (--vfm)       : radiov3 / radiov2.5 / dinov2 / dinov3 / pe (implementations in vfms/)

To evaluate the finished bank: set _mb_bank in configs/base_config.py to the bank name, then run
    python eval.py --config configs/cfg_voc21.py
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent

CLIP_CHOICES = ('meta_b', 'meta_l', 'dfn_b', 'dfn_l')
VFM_CHOICES = ('radiov3', 'radiov2.5', 'dinov2', 'dinov3', 'pe')   # keep in sync with vfms.VFM_REGISTRY

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--img', required=True, help='Image directory (scanned recursively)')
parser.add_argument('--data_name', required=True, help='Bank name (baked into all output filenames; set _mb_bank in the eval config to it)')
parser.add_argument('--mask_model', default='entityseg', choices=('entityseg', 'eomt', 'sam', 'sam2'))
parser.add_argument('--clip', nargs='+', default=['meta_b', 'dfn_b'], choices=CLIP_CHOICES,
                    help='CLIP feature banks to generate (default: primary meta_b + auxiliary dfn_b, as deployed in the paper)')
parser.add_argument('--vfm', default='radiov3', choices=VFM_CHOICES,
                    help='VFM used for the retrieval index (generate_vfm_embeddings --model); must match vfm_type in the deployment config')
parser.add_argument('--slide_short', default=448, type=int)
parser.add_argument('--slide_crop', default=336, type=int)
parser.add_argument('--gpus', default=None, help='e.g. "0,1"; unset keeps the current CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

args.img = os.path.abspath(args.img)
REGION_SET = args.data_name


def run(script, *a):
    cmd = [sys.executable, str(_DIR / script), *map(str, a)]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    if args.gpus is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpus
    r = subprocess.run(cmd, env=env, cwd=_ROOT)
    if r.returncode != 0:
        print(f"\nStep failed ({script}, exit code {r.returncode}); pipeline stopped. Fix and rerun the same command to resume.")
        sys.exit(r.returncode)


# 1) Instance masks
run('generate_mask.py', '--model', args.mask_model, '--data_name', args.data_name, '--image_dir', args.img)

# 2) Manifest (hard prerequisite of the embedding scripts)
run('build_manifest.py', '--data_name', args.data_name, '--mask_model', args.mask_model, '--img', args.img)
manifest_tsv = _ROOT / 'memory_bank' / 'manifests' / f'{REGION_SET}.images.tsv'
if not manifest_tsv.exists() or manifest_tsv.stat().st_size == 0:
    print(f"Manifest generation failed or is empty: {manifest_tsv}; aborting.")
    sys.exit(1)

# 3) CLIP region-feature banks (primary + auxiliary)
for clip in args.clip:
    run('generate_clip_embeddings.py', '--clip', clip, '--data_name', args.data_name,
        '--mask_model', args.mask_model, '--img', args.img,
        '--slide_short', args.slide_short, '--slide_crop', args.slide_crop)

# 4) VFM region features + FAISS retrieval index
run('generate_vfm_embeddings.py', '--model', args.vfm, '--data_name', args.data_name,
    '--mask_model', args.mask_model, '--img', args.img)

print('\n================ Memory bank build complete ================')
print(f'REGION_SET = {REGION_SET}')
for sub in ('clip_embeddings', 'vfm_embeddings'):
    d = _ROOT / 'memory_bank' / sub
    for f in sorted(d.glob(f'{REGION_SET}_*')):
        if f.suffix in ('.pt', '.faiss'):
            print(f'  {f.relative_to(_ROOT)}  ({f.stat().st_size / 2**20:.0f} MiB)')
print('\nTo evaluate (voc21 as an example): set _mb_bank in configs/base_config.py to '
      f"'{REGION_SET}', then run")
print('  python eval.py --config configs/cfg_voc21.py')
