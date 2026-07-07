"""Shared benchmark utilities: sample generators, timers, stats."""
import gc
import os
import time
import statistics
import pathlib

import torch


def make_sample(kind: str, index: int):
    """Return a representative sample of the requested size class.

    small  : a dict of scalars + a tiny tensor  (~a few hundred bytes)
    medium : a ~1 MB float32 tensor
    large  : a ~50 MB float32 tensor
    """
    if kind == "small":
        return {
            "id": index,
            "label": index % 10,
            "score": float(index) * 1.5,
            "vec": torch.arange(16, dtype=torch.float32) + index,
        }
    if kind == "medium":
        # 262144 float32 = 1 MiB
        return torch.full((262144,), float(index), dtype=torch.float32)
    if kind == "large":
        # 13_107_200 float32 = 50 MiB
        return torch.full((13_107_200,), float(index), dtype=torch.float32)
    raise ValueError(kind)


DEFAULT_N = {"small": 500, "medium": 100, "large": 10}


def time_calls(fn, n):
    """Call fn(i) for i in range(n); return (total_seconds, per_call_latencies)."""
    lat = []
    t0 = time.perf_counter()
    for i in range(n):
        s = time.perf_counter()
        fn(i)
        lat.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    return total, lat


def latency_stats(lat):
    lat_ms = sorted(x * 1e3 for x in lat)
    n = len(lat_ms)

    def pct(p):
        if n == 0:
            return 0.0
        k = min(n - 1, int(round(p / 100 * (n - 1))))
        return lat_ms[k]

    return {
        "mean_ms": statistics.fmean(lat_ms) if lat_ms else 0.0,
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p99_ms": pct(99),
        "max_ms": lat_ms[-1] if lat_ms else 0.0,
    }


def dir_size_bytes(path):
    p = pathlib.Path(path)
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def rss_bytes():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return 0


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def clear_os_cache_hint():
    """Best-effort: drop refs and gc so warm-read isn't measuring gc."""
    gc.collect()
