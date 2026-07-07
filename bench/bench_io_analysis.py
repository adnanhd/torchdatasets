"""I/O inefficiency microbenchmarks.

Investigates, with numbers:
  1. pickle protocol 2 (Tensor default) vs 4 vs 5 (default)
  2. torch.save vs raw pickle vs numpy.save/load
  3. buffered vs unbuffered writes
  4. flush/fsync cost
  5. redundant stat() in __contains__ (stat then reopen)
  6. mmap warm reads (numpy memmap, torch.load(mmap=True))
"""
import io
import json
import os
import pathlib
import pickle
import shutil
import time

import numpy as np
import torch

SCRATCH = pathlib.Path(__file__).parent / "_scratch_io"


def timeit(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def bench_protocols():
    """Serialize a 1MB tensor with torch.save at each pickle protocol."""
    t = torch.randn(262144)  # 1 MiB
    out = {}
    for proto in (2, 4, 5):
        buf = io.BytesIO()
        torch.save(t, buf, pickle_protocol=proto)
        size = len(buf.getvalue())
        w = timeit(lambda: torch.save(t, io.BytesIO(), pickle_protocol=proto), 200)
        data = buf.getvalue()
        r = timeit(lambda: torch.load(io.BytesIO(data), weights_only=False), 200)
        out[f"proto{proto}"] = {"write_ms": w * 1e3, "read_ms": r * 1e3, "bytes": size}
    return out


def bench_backends():
    """torch.save vs raw pickle vs numpy.save for a 1MB tensor, to disk."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t = torch.randn(262144)
    arr = t.numpy()
    out = {}

    p = SCRATCH / "t.pt"
    out["torch.save_p2"] = {
        "write_ms": timeit(lambda: torch.save(t, p, pickle_protocol=2), 200) * 1e3,
        "read_ms": timeit(lambda: torch.load(p, weights_only=False), 200) * 1e3,
        "bytes": p.stat().st_size,
    }
    out["torch.save_p5"] = {
        "write_ms": timeit(lambda: torch.save(t, p, pickle_protocol=5), 200) * 1e3,
        "read_ms": timeit(lambda: torch.load(p, weights_only=False), 200) * 1e3,
        "bytes": p.stat().st_size,
    }

    def raw_pickle_write():
        with open(SCRATCH / "t.pkl", "wb") as f:
            pickle.dump(t, f, protocol=2)

    def raw_pickle_write5():
        with open(SCRATCH / "t.pkl", "wb") as f:
            pickle.dump(t, f, protocol=5)

    def raw_pickle_read():
        with open(SCRATCH / "t.pkl", "rb") as f:
            return pickle.load(f)

    out["pickle_p2"] = {"write_ms": timeit(raw_pickle_write, 200) * 1e3}
    out["pickle_p2"]["read_ms"] = timeit(raw_pickle_read, 200) * 1e3
    out["pickle_p2"]["bytes"] = (SCRATCH / "t.pkl").stat().st_size
    out["pickle_p5"] = {"write_ms": timeit(raw_pickle_write5, 200) * 1e3}
    out["pickle_p5"]["read_ms"] = timeit(raw_pickle_read, 200) * 1e3
    out["pickle_p5"]["bytes"] = (SCRATCH / "t.pkl").stat().st_size

    npy = SCRATCH / "t.npy"
    out["numpy.save"] = {
        "write_ms": timeit(lambda: np.save(npy, arr), 200) * 1e3,
        "read_ms": timeit(lambda: np.load(npy), 200) * 1e3,
        "bytes": npy.stat().st_size,
    }
    out["numpy.load_mmap"] = {
        "read_ms": timeit(lambda: np.load(npy, mmap_mode="r")[0], 500) * 1e3,
    }
    return out


def bench_fsync():
    """Cost of flush+fsync vs plain buffered write for a 1MB payload."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t = torch.randn(262144)
    payload = pickle.dumps(t, protocol=5)
    out = {}

    def plain():
        with open(SCRATCH / "s.bin", "wb") as f:
            f.write(payload)

    def with_fsync():
        with open(SCRATCH / "s.bin", "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

    def unbuffered():
        with open(SCRATCH / "s.bin", "wb", buffering=0) as f:
            f.write(payload)

    out["buffered"] = timeit(plain, 200) * 1e3
    out["buffered_fsync"] = timeit(with_fsync, 100) * 1e3
    out["unbuffered"] = timeit(unbuffered, 200) * 1e3
    return out


def bench_stat_overhead():
    """Cost of the __contains__ stat() that precedes every cached read."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    p = SCRATCH / "exists.pt"
    torch.save(torch.randn(262144), p)
    import pathlib as _pl

    def contains_via_pathlib():
        return _pl.Path(str(p)).is_file()

    def contains_via_os():
        return os.path.exists(p)

    def full_load():
        return torch.load(p, weights_only=False)

    out = {
        "pathlib_is_file_us": timeit(contains_via_pathlib, 5000) * 1e6,
        "os_path_exists_us": timeit(contains_via_os, 5000) * 1e6,
        "torch_load_us": timeit(full_load, 200) * 1e6,
    }
    out["stat_pct_of_load"] = 100 * out["pathlib_is_file_us"] / out["torch_load_us"]
    return out


def bench_mmap_warm():
    """Warm-read latency: torch.load full vs torch.load(mmap=True) touching a slice."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t = torch.randn(13_107_200)  # 50 MiB
    p = SCRATCH / "big.pt"
    torch.save(t, p)
    out = {}
    out["torch_load_full_ms"] = timeit(lambda: torch.load(p, weights_only=False), 30) * 1e3
    try:
        out["torch_load_mmap_ms"] = timeit(
            lambda: torch.load(p, weights_only=False, mmap=True)[0].item(), 30) * 1e3
    except TypeError:
        out["torch_load_mmap_ms"] = None  # older torch without mmap kwarg
    return out


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    results = {
        "protocols_1MB_tensor": bench_protocols(),
        "backends_1MB_tensor": bench_backends(),
        "fsync_1MB": bench_fsync(),
        "stat_overhead": bench_stat_overhead(),
        "mmap_warm_50MB": bench_mmap_warm(),
    }
    shutil.rmtree(SCRATCH, ignore_errors=True)
    out = pathlib.Path(__file__).parent / "results_io.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
