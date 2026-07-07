"""Benchmark prototype cachers vs the stock ones.

Warm-read note: MmapTensor/ShardedTensor return lazily-mapped tensors, so
we report two warm-read numbers: 'lazy' (return object only, as __getitem__
does) and 'touch' (force full materialization via .sum()), so mmap's
zero-copy advantage isn't overstated.
"""
import argparse
import json
import pathlib
import shutil
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import (make_sample, DEFAULT_N, time_calls, latency_stats,
                    dir_size_bytes, human, clear_os_cache_hint)
import torchdatasets as td
import prototypes as proto

SCRATCH = pathlib.Path(__file__).parent / "_scratch_proto"


def touch(x):
    if torch.is_tensor(x):
        return float(x.sum())
    if isinstance(x, dict):
        return sum(float(v.sum()) if torch.is_tensor(v) else 0 for v in x.values())
    return 0


def bench(factory, name, kind, n, do_touch):
    path = SCRATCH / f"{name}_{kind}"
    if path.exists():
        shutil.rmtree(path)
    samples = [make_sample(kind, i) for i in range(n)]
    c = factory(path)
    w_total, w_lat = time_calls(lambda i: c.__setitem__(i, samples[i]), n)
    del samples
    clear_os_cache_hint()
    r_total, r_lat = time_calls(lambda i: c[i], n)
    disk = dir_size_bytes(path)  # after reads: ShardedTensor's buffer is flushed
    touch_total = None
    if do_touch:
        clear_os_cache_hint()
        touch_total, _ = time_calls(lambda i: touch(c[i]), n)
    if hasattr(c, "clean"):
        c.clean()
    shutil.rmtree(path, ignore_errors=True)
    return {
        "cacher": name, "kind": kind, "n": n,
        "write_sps": n / w_total, "read_lazy_sps": n / r_total,
        "read_touch_sps": (n / touch_total) if touch_total else None,
        "write_lat": latency_stats(w_lat), "read_lat": latency_stats(r_lat),
        "disk_bytes": disk,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["small", "medium", "large"])
    args = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)

    variants = [
        (lambda p: td.cachers.Pickle(p), "Pickle(orig)", False),
        (lambda p: proto.PickleP5(p), "PickleP5", False),
        (lambda p: td.cachers.Tensor(p), "Tensor(orig,p2)", False),
        (lambda p: proto.TensorP5(p), "TensorP5", False),
        (lambda p: proto.MmapTensor(p), "MmapTensor", True),
        (lambda p: proto.ShardedTensor(p), "ShardedTensor", True),
    ]
    results = []
    for kind in args.kinds:
        n = DEFAULT_N[kind]
        print(f"== {kind} (n={n}) ==", flush=True)
        for factory, name, do_touch in variants:
            r = bench(factory, name, kind, n, do_touch)
            results.append(r)
            rt = f"{r['read_touch_sps']:8.1f}" if r['read_touch_sps'] else "     n/a"
            print(f"  {name:16s} write {r['write_sps']:9.1f}  read_lazy {r['read_lazy_sps']:9.1f}  "
                  f"read_touch {rt}  disk {human(r['disk_bytes'])}", flush=True)

    out = pathlib.Path(__file__).parent / "results_proto.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
