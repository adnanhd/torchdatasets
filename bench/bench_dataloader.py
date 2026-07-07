"""Investigate the Memory-cacher footgun under DataLoader(num_workers>0).

The Memory cacher stores samples in a plain in-process dict. DataLoader
workers are separate processes. Whatever a worker caches lives only in
that worker's address space; it never propagates to the parent, and with
non-persistent workers it is discarded when the worker dies at epoch end.

Consequences:
  - the parent-process cache stays EMPTY forever
  - with persistent_workers=False every worker recomputes every sample
    it is handed, every epoch: the cache never warms
  - with persistent_workers=True a worker warms only its OWN private
    slice, and only for indices it happens to be assigned again

We demonstrate this two ways that are robust to the mp start method:
  (1) wall-clock per epoch (a warm cache makes later epochs ~free)
  (2) inspecting len(parent_cache) after all epochs (should be full if
      caching worked; it stays 0 with workers)
"""
import multiprocessing as mp
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torchdatasets as td

SLEEP = 0.002  # simulated expensive base __getitem__


class SlowDataset(td.Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        time.sleep(SLEEP)  # only paid on a cache MISS
        return torch.tensor([float(index)])


def run(num_workers, persistent, epochs=3, n=64, shared=False):
    ds = SlowDataset(n)
    if shared:
        mgr = mp.Manager()
        ds.cache(td.cachers.Memory(cache=mgr.dict()))
    else:
        ds.cache()  # default Memory cacher
    kw = {}
    if num_workers > 0:
        kw["persistent_workers"] = persistent
    dl = DataLoader(ds, batch_size=8, num_workers=num_workers, shuffle=False, **kw)
    per_epoch = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        for _ in dl:
            pass
        per_epoch.append(time.perf_counter() - t0)
    parent_cache_size = len(ds._cachers[0].cache)
    return per_epoch, parent_cache_size


def main():
    print("mp start method:", mp.get_start_method())
    n = 64
    ideal_cold = n * SLEEP
    print(f"n={n}, sleep/miss={SLEEP*1e3:.0f}ms -> a cold full epoch ~= {ideal_cold*1e3:.0f}ms\n")
    print("A working cache: epoch1 ~cold, epoch2/3 ~free, parent_cache=64/64.\n")

    configs = [
        ("num_workers=0", 0, False, False),
        ("num_workers=2, persistent=False", 2, False, False),
        ("num_workers=2, persistent=True", 2, True, False),
        ("num_workers=4, persistent=True", 4, True, False),
        ("num_workers=4, Manager().dict()", 4, False, True),
    ]
    for label, nw, persist, shared in configs:
        per_epoch, parent_size = run(nw, persist, shared=shared)
        ms = [f"{x*1e3:6.0f}" for x in per_epoch]
        warmed = per_epoch[-1] < ideal_cold * 0.3
        print(f"{label:34s} epochs(ms)=[{', '.join(ms)}]  "
              f"parent_cache={parent_size:2d}/{n}  later_epochs_warm={'YES' if warmed else 'NO'}")


if __name__ == "__main__":
    main()
