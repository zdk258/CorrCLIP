"""Slot-based memmap vector-store writer: multiple GPU processes concurrently write their own rows keyed by manifest id; once all slots are filled the store is finalized into a single .pt.

Intermediate files (exist during the build, deleted after finalize; rerunning the same command resumes):
  {BASE}.f16.npy   [N, dim] fp16 (standard np.lib.format .npy, readable via np.load(mmap_mode))
  {BASE}.done.npy  [N] bool, whether each slot has been written
Final product:
  {BASE}.pt        torch.save'd [N, dim] fp16 tensor, row index = manifest id (read directly by the eval-side PTTensorDataset)

Concurrency safety: the manifest pre-assigns every region a unique id (row index); each process writes only the rows of its own images, the row sets are pairwise disjoint, so concurrent memmap writes need no locking.
"""
import os

import numpy as np
import torch


class PtSlotWriter:
    def __init__(self, base_path, n, dim):
        self.base = base_path
        self.data_path = base_path + '.f16.npy'
        self.done_path = base_path + '.done.npy'
        self.pt_path = base_path + '.pt'
        self.n, self.dim = int(n), int(dim)
        self._data = None
        self._done = None

    # ---- Main process: create/validate intermediate files (existing files -> resume; an existing finalized .pt has its rows imported as a prefix = incremental extension) ----
    def create(self):
        seed = None
        if os.path.exists(self.pt_path) and not os.path.exists(self.data_path):
            seed = torch.load(self.pt_path, map_location='cpu')
            if seed.shape[0] > self.n or seed.dim() != 2 or seed.shape[1] != self.dim:
                raise RuntimeError(f'{self.pt_path} shape {tuple(seed.shape)} conflicts with the manifest-expected [{self.n},{self.dim}]'
                                   f' (the manifest only grows; delete that .pt first to rebuild).')
        if not os.path.exists(self.data_path):
            np.lib.format.open_memmap(self.data_path, mode='w+', dtype=np.float16, shape=(self.n, self.dim)).flush()
        if not os.path.exists(self.done_path):
            np.lib.format.open_memmap(self.done_path, mode='w+', dtype=bool, shape=(self.n,)).flush()
        d = np.load(self.data_path, mmap_mode='r')
        if d.shape != (self.n, self.dim) or d.dtype != np.float16:
            raise RuntimeError(f'{self.data_path}: shape {d.shape}/{d.dtype} does not match the expected [{self.n},{self.dim}] fp16'
                               f' (manifest changed? delete the intermediate files and rerun).')
        if seed is not None:
            k = seed.shape[0]
            data = np.load(self.data_path, mmap_mode='r+')
            data[:k] = seed.numpy()
            data.flush()
            done = np.load(self.done_path, mmap_mode='r+')
            done[:k] = True
            done.flush()
            print(f'[pt_store] Incremental extension: imported the first {k} rows from {self.pt_path}; {self.n - k} new slots.')

    def done_mask(self):
        return np.array(np.load(self.done_path, mmap_mode='r'))

    # ---- Worker processes: open, write, flush ----
    def open_for_write(self):
        self._data = np.load(self.data_path, mmap_mode='r+')
        self._done = np.load(self.done_path, mmap_mode='r+')

    def write(self, ids, vecs):
        """ids: iterable of int; vecs: [k, dim] torch tensor or np array, ordered to match ids."""
        ids = np.asarray(list(ids), dtype=np.int64)
        arr = vecs.detach().cpu().numpy() if torch.is_tensor(vecs) else np.asarray(vecs)
        self._data[ids] = arr.astype(np.float16, copy=False)
        self._done[ids] = True

    def flush(self):
        self._data.flush()
        self._done.flush()

    # ---- Main process: finalize into .pt once all slots are done ----
    def finalize(self):
        done = self.done_mask()
        n_done = int(done.sum())
        if n_done < self.n:
            print(f'[pt_store] {self.base}: {n_done}/{self.n} slots written, not yet complete; not finalizing (rerun the same command to resume).')
            return False
        data = torch.from_numpy(np.array(np.load(self.data_path, mmap_mode='r')))
        torch.save(data, self.pt_path)
        os.remove(self.data_path)
        os.remove(self.done_path)
        print(f'[pt_store] Finalized: {self.pt_path}  [{self.n}, {self.dim}] fp16')
        return True
