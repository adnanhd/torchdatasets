"""Per-__getitem__ Python overhead of the framework on a warm hit,
plus the cost of the redundant contains()+getitem() double-lookup and a
merged get()-based alternative.
"""
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torchdatasets as td


def timeit(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e9  # ns


class DS(td.Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return torch.tensor([float(index)])


def main():
    n = 64
    ds = DS(n).cache()
    for i in range(n):
        _ = ds[i]  # warm the Memory cache
    raw = ds._cachers[0].cache  # the underlying dict

    reps = 500_000
    idx = 7
    print("Warm-hit read, ns/call (n=64 cached samples):")
    print(f"  bare dict[idx]                 : {timeit(lambda: raw[idx], reps):8.1f}")
    print(f"  dict __contains__ then __getitem__ (framework pattern):")
    print(f"    (idx in raw); raw[idx]       : {timeit(lambda: (idx in raw, raw[idx]), reps):8.1f}")
    print(f"  dict.get(idx) merged lookup    : {timeit(lambda: raw.get(idx), reps):8.1f}")
    print(f"  td.Dataset.__getitem__(idx)    : {timeit(lambda: ds[idx], reps):8.1f}")

    # framework overhead factor
    bare = timeit(lambda: raw[idx], reps)
    full = timeit(lambda: ds[idx], reps)
    print(f"\n  framework wrapper overhead: {full/bare:.1f}x a bare dict lookup "
          f"(+{full-bare:.0f} ns/call)")


if __name__ == "__main__":
    main()
