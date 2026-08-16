"""Export a legacy LMDB feature bank into a single .pt file (the input format of PTTensorDataset).

The export is an [N, D] fp16 tensor whose row index = the LMDB's 8-digit decimal key.
Values are bit-identical to the LMDB records; by default a random sample is re-verified bit-for-bit after export.

Usage:
    python memory_bank_builder/lmdb_to_pt.py memory_bank/clip_embeddings/{BASE}.lmdb [--out xxx.pt]
"""
import argparse
import time

import lmdb
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lmdb_path')
    ap.add_argument('--out', default=None, help='Output path; defaults to the input name with .lmdb replaced by .pt')
    ap.add_argument('--check', type=int, default=2000, help='After export, randomly sample N records and verify bit-for-bit against the LMDB (0 disables)')
    args = ap.parse_args()

    out = args.out or (args.lmdb_path[:-5] + '.pt' if args.lmdb_path.endswith('.lmdb')
                       else args.lmdb_path + '.pt')

    t0 = time.time()
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=True)
    with env.begin() as txn:
        n = txn.stat()['entries']
        cur = txn.cursor()
        assert cur.first(), f'{args.lmdb_path} is empty'
        dim = len(cur.value()) // 2                       # fp16
        assert cur.last()
        max_id = int(cur.key().decode('ascii'))
        assert max_id == n - 1, f'ids not contiguous: entries={n} but max key={max_id}'

        arr = np.empty((n, dim), dtype=np.float16)
        filled = 0
        for k, v in txn.cursor():
            arr[int(k.decode('ascii'))] = np.frombuffer(v, dtype=np.float16)
            filled += 1
            if filled % 500000 == 0:
                print(f'  {filled}/{n} ({time.time() - t0:.0f}s)')
        assert filled == n
    data = torch.from_numpy(arr)
    torch.save(data, out)
    print(f'[lmdb_to_pt] {args.lmdb_path} -> {out}  [{n}, {dim}] fp16  ({time.time() - t0:.0f}s)')

    if args.check:
        rng = np.random.default_rng(0)
        ids = rng.choice(n, size=min(args.check, n), replace=False)
        with env.begin() as txn:
            for i in ids:
                v = txn.get(f'{i:08}'.encode('ascii'))
                assert v == data[int(i)].numpy().tobytes(), f'id {i} mismatch'
        print(f'[lmdb_to_pt] Sampled {len(ids)} records: bit-for-bit identical')
    env.close()


if __name__ == '__main__':
    main()
