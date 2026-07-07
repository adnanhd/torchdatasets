"""Benchmark the three stock cachers: Memory, Pickle, Tensor.

Measures cold write throughput, warm read throughput, per-sample latency
distribution, peak RSS (Memory) and disk footprint (Pickle/Tensor).
"""
import argparse
import json
import pathlib
import shutil
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import (make_sample, DEFAULT_N, time_calls, latency_stats,
                    dir_size_bytes, rss_bytes, human, clear_os_cache_hint)

import torchdatasets as td

SCRATCH = pathlib.Path(__file__).parent / "_scratch"


def bench_memory(kind, n):
    # Measure RSS of what the cache *retains*: generate fresh inside the
    # write loop so only the cacher dict holds references.
    cacher = td.cachers.Memory()
    clear_os_cache_hint()
    rss0 = rss_bytes()
    write_total, write_lat = time_calls(
        lambda i: cacher.__setitem__(i, make_sample(kind, i)), n)
    rss1 = rss_bytes()
    read_total, read_lat = time_calls(lambda i: cacher[i], n)
    return {
        "cacher": "Memory", "kind": kind, "n": n,
        "cold_write_s": write_total, "warm_read_s": read_total,
        "write_throughput_sps": n / write_total,
        "read_throughput_sps": n / read_total,
        "write_lat": latency_stats(write_lat),
        "read_lat": latency_stats(read_lat),
        "rss_delta_bytes": rss1 - rss0,
        "disk_bytes": 0,
    }


def bench_disk(cacher_factory, name, kind, n):
    path = SCRATCH / f"{name}_{kind}"
    if path.exists():
        shutil.rmtree(path)
    samples = [make_sample(kind, i) for i in range(n)]
    cacher = cacher_factory(path)

    write_total, write_lat = time_calls(lambda i: cacher.__setitem__(i, samples[i]), n)
    disk = dir_size_bytes(path)

    # contains-check cost (the redundant stat in the hot path)
    contains_total, contains_lat = time_calls(lambda i: (i in cacher), n)

    del samples
    clear_os_cache_hint()
    read_total, read_lat = time_calls(lambda i: cacher[i], n)

    shutil.rmtree(path, ignore_errors=True)
    return {
        "cacher": name, "kind": kind, "n": n,
        "cold_write_s": write_total, "warm_read_s": read_total,
        "write_throughput_sps": n / write_total,
        "read_throughput_sps": n / read_total,
        "contains_total_s": contains_total,
        "contains_lat": latency_stats(contains_lat),
        "write_lat": latency_stats(write_lat),
        "read_lat": latency_stats(read_lat),
        "rss_delta_bytes": 0,
        "disk_bytes": disk,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["small", "medium", "large"])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "results_cachers.json"))
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    results = []
    for kind in args.kinds:
        n = DEFAULT_N[kind]
        print(f"== {kind} (n={n}) ==", flush=True)
        for r in (
            bench_memory(kind, n),
            bench_disk(lambda p: td.cachers.Pickle(p), "Pickle", kind, n),
            bench_disk(lambda p: td.cachers.Tensor(p), "Tensor", kind, n),
        ):
            results.append(r)
            print(f"  {r['cacher']:8s} write {r['write_throughput_sps']:9.1f} sps  "
                  f"read {r['read_throughput_sps']:9.1f} sps  "
                  f"disk {human(r['disk_bytes'])}  rss {human(r['rss_delta_bytes'])}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
