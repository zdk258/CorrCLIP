"""
Region ID registry (manifest): an append-only mapping from each (image, instance)
region to a stable integer id.

The id replaces the old "position in a global sort", which fundamentally:
  - removes the final sort (each vector lands directly in its id slot at
    extraction time);
  - enables incremental builds (new images only append new ids; existing ids
    never change);
  - enables subsets (ids are laid out contiguously per image, so "first N
    images" / "first N embeddings" are both id prefix ranges).

Files (named after the bank, under memory_bank/manifests/):
  {bank}.manifest.txt   line number (0-based) = id, content = '{stem}#{idx}'
  {bank}.images.tsv     one line per image: '{stem}\t{start_id}\t{num_regions}',
                        in image arrival order

Rules: append image by image; each image's region ids are contiguous. An
incremental build appends only not-yet-registered images, sorted by stem, to
the end.
"""
import os
import multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def paths(manifests_dir, region_set):
    base = os.path.join(manifests_dir, region_set)
    return base + '.manifest.txt', base + '.images.tsv'


def num_regions(mask_path):
    """Number of instances (regions) in this mask, consistent with the torch.unique enumeration on the producing side."""
    with np.load(mask_path) as data:
        m = data['instance_mask']
    if m.ndim == 2:
        u = np.unique(m)
        return int((u != 0).sum())
    return int(m.shape[0])  # 3D bool: dim 0 indexes instances


def _count_one(args):
    """Process-pool worker: return (stem, num_regions); must stay picklable."""
    stem, mask_dir = args
    return stem, num_regions(os.path.join(mask_dir, stem + '.npz'))


def _iter_counts(new_stems, mask_dir, workers):
    """Yield (stem, n) in new_stems order; with workers>1, read the npz files in parallel via a process pool."""
    if workers <= 1:
        for stem in new_stems:
            yield stem, num_regions(os.path.join(mask_dir, stem + '.npz'))
        return
    with mp.Pool(workers) as pool:
        # imap preserves order: writes hit disk in the same order as the serial version, so ids stay deterministic
        yield from pool.imap(_count_one, ((s, mask_dir) for s in new_stems), chunksize=64)


def load_region_keys(manifest_txt):
    """Line number (0-based) = id; return ['{stem}#{idx}', ...]."""
    if not os.path.exists(manifest_txt):
        return []
    with open(manifest_txt) as f:
        return f.read().splitlines()


def load_images(images_tsv):
    """Return [(stem, start_id, num_regions), ...] in image arrival order."""
    out = []
    if os.path.exists(images_tsv):
        with open(images_tsv) as f:
            for line in f.read().splitlines():
                if not line:
                    continue
                stem, start, n = line.split('\t')
                out.append((stem, int(start), int(n)))
    return out


def stem_to_start(images):
    """{stem: start_id}, so a worker can derive id = start_id + idx from (stem, idx)."""
    return {stem: start for stem, start, _ in images}


def total_regions(images):
    return sum(n for _, _, n in images)


def build_or_extend(manifests_dir, region_set, image_dir, mask_dir, workers=None):
    """Incrementally build/extend the manifest from image_dir (recursive) ∩ mask_dir.

    Appends only images that are in image_dir, have a mask, and are not yet
    registered (sorted by stem within the batch for determinism). Existing ids
    never change. Returns (region_keys, images).

    workers: number of parallel processes for reading/counting mask regions;
    None = auto (min(32, CPU count)), 1 = serial.
    """
    os.makedirs(manifests_dir, exist_ok=True)
    manifest_txt, images_tsv = paths(manifests_dir, region_set)

    region_keys = load_region_keys(manifest_txt)
    images = load_images(images_tsv)
    known = {stem for stem, _, _ in images}
    next_id = len(region_keys)

    # Find new images in image_dir that have a mask but are not yet registered
    new_stems, seen = [], set()
    for root, _, files in tqdm(os.walk(image_dir), desc='Scanning image dir', unit='dir'):
        for name in files:
            if not name.lower().endswith(IMG_EXTS):
                continue
            stem = Path(name).stem
            if stem in known or stem in seen:
                continue
            if os.path.exists(os.path.join(mask_dir, stem + '.npz')):
                seen.add(stem)
                new_stems.append(stem)
    new_stems.sort()  # determinism within the batch

    if workers is None:
        workers = min(32, os.cpu_count() or 4)

    appended_imgs = []
    counts = _iter_counts(new_stems, mask_dir, workers)
    with open(manifest_txt, 'a') as fm, open(images_tsv, 'a') as fi:
        for stem, n in tqdm(counts, total=len(new_stems),
                            desc=f'{region_set} (x{workers})', unit='img'):
            if n == 0:                      # all-background mask yields no regions, matching the producing side
                continue
            fi.write(f'{stem}\t{next_id}\t{n}\n')
            for idx in range(n):
                fm.write(f'{stem}#{idx}\n')
            appended_imgs.append((stem, next_id, n))
            region_keys.extend(f'{stem}#{idx}' for idx in range(n))
            next_id += n

    images.extend(appended_imgs)
    return region_keys, images
