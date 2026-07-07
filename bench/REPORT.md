# torchdatasets caching / I/O performance report

Stress-test and optimization survey of the caching layer
(`torchdatasets/cachers.py`, `torchdatasets/datasets.py`, `torchdatasets/_base.py`).

Run the harness with `python bench/bench_cachers.py` (and the other `bench_*.py`
scripts) against an installed copy of the package; raw numbers land in
`bench/results_*.json`.

## Upstream status

Applied to `torchdatasets/cachers.py` from this survey:

- **`pickle.HIGHEST_PROTOCOL` by default** in `Tensor` (was hardcoded protocol 2)
  and `Pickle` - the single biggest, free I/O win. Uses protocol 5 on Python
  >= 3.8 and protocol 4 on 3.7, so it stays valid on the whole support range.
- **`os.path.exists` in `__contains__`** instead of `pathlib.Path.is_file`.
- **`Memory` + `DataLoader(num_workers>0)` footgun documented** in the docstring.
- **`weights_only=False` compatibility guard** in `Tensor` so arbitrary cached
  samples still load on torch >= 2.6 (which flipped that default to `True`).
- **New opt-in cachers**: `MmapTensor` (zero-copy warm reads via
  `torch.load(mmap=True)`, gated to torch >= 2.1) and `Sharded` (single-file
  LMDB-style store for many small samples).
- **Hot-path micro-opts** in `_dev_utils.py`: `apply_mapping` indexes the map
  list in place with an empty-range short-circuit instead of slicing a copy on
  every call, and `reversed_enumerate` is a hand-rolled generator (about 2x the
  `zip(range, reversed)` form). Measured end to end, warm `Dataset.__getitem__`
  drops from ~480 ns to ~360 ns per sample, roughly 1.3x.

Deliberately **not** adopted:

- **Merged cache lookup** (replacing `index in cacher` then `cacher[index]` with
  a single `.get()`): only ~1.09x on a dict and it would change the public
  `Cacher` contract that custom cachers implement. Not worth it.

- **Cython** - a pure-python `cythonize` of the hot path builds cleanly but only
  yields ~1.24x untyped, which does not justify a compiled build step and the
  wheel/packaging complexity it adds. See `bench/bench_cython.py`.

## Environment
- CPU-only PyTorch 2.12.1+cpu, Python 3.14, NumPy 2.5, Cython 3.2.8
- Local scratch FS, warm page cache (not cold-disk / network-FS numbers).
- mp start method: forkserver (Python 3.14 default on Linux - matters for the DataLoader footgun).

## How the cache is wired
`_base.MetaDataset` wraps every `Dataset.__getitem__`. On each access it walks the cachers
newest-first and per cacher does TWO calls: `if index in cacher` (`__contains__` -> stat() for
disk cachers) then `cacher[index]` (`__getitem__` -> open()+read() again). So a disk cache hit
touches the file twice. `apply_mapping` slices `mappings[start:end]` (list copy) every call even
with no maps; `reversed_enumerate` rebuilds range+reversed+zip objects every call.

## Methodology
small = dict of scalars + 16-elt tensor (~hundreds of bytes, n=500); medium = 1 MiB float32 (n=100);
large = 50 MiB float32 (n=10). Cold write = first-epoch populate; warm read = re-read all. RSS via
psutil, disk via stat. Harness: bench/*.py; raw numbers in bench/results_*.json.
Caveat: warm page cache means disk-cacher READ numbers reflect (de)serialization, not physical I/O;
on cold/networked storage disk cachers get much slower and the mmap/sharded wins grow.

## 1. Stock cacher results (throughput sps)
Memory write includes sample generation (undersells; pure dict-insert ~5.2M sps); its read is clean.

| kind | cacher | cold write | warm read | disk | Memory RSS |
|------|--------|-----------:|----------:|------|-----------:|
| small  | Memory | (gen-bound) | 3,894,051 | -      | 3.7 MB |
| small  | Pickle | 18,167     | 18,067    | 247 KB | - |
| small  | Tensor | 16,186     | 10,521    | 780 KB | - |
| medium | Memory | (gen-bound) | 4,503,289 | -     | 101 MB |
| medium | Pickle | 3,069      | 1,466     | 100 MB | - |
| medium | Tensor | 2,157      | 4,575     | 100 MB | - |
| large  | Memory | (gen-bound) | 2,073,861 | -     | 450 MB |
| large  | Pickle | 37.3       | 42.7      | 500 MB | - |
| large  | Tensor | 49.6       | 53.1      | 500 MB | - |

- Memory is 100-4000x faster than any disk cacher, but RSS == full dataset size (450 MB retained for
  500 MB of tensors; stored by reference, ~zero overhead but unbounded). RAM gamble, not a scaling plan.
- Pickle is the worst tensor cacher: medium read 1,466 vs Tensor 4,575 (3.1x slower).
- Tensor's disk footprint for small dicts is 3.2x Pickle's (780 KB vs 247 KB) - torch.save ZIP overhead.

## 2. I/O inefficiency analysis
### 2a. Pickle protocol - biggest lever (1 MiB tensor)
| method | write ms | read ms | bytes |
|--------|---------:|--------:|------:|
| raw pickle proto 2 | 18.81 | 5.53 | 1,579,114 |
| raw pickle proto 5 | 9.73  | 0.24 | 1,048,988 |
| torch.save proto 2 | 6.85  | 0.34 | 1,049,919 |
| torch.save proto 5 | 8.64  | 0.19 | 1,049,919 |
| numpy.save         | 8.05  | 0.15 | 1,048,704 |
| numpy.load(mmap)   | -     | 0.059| - |

- Raw pickle proto 2 -> 5: read 23x faster (5.53 -> 0.24 ms), file 33% smaller. (This is the Pickle cacher.)
- torch.save proto 2 -> 5: read 1.8x faster (0.34 -> 0.19). Write marginally slower; reads happen every epoch.
- numpy.save is most compact + fastest full read; mmap another ~2.5x on top.

### 2b. fsync / buffering (1 MiB)
buffered 8.85 ms; buffered+fsync 34.43 ms; unbuffered 7.38 ms. Neither cacher fsyncs - correct; adding it costs ~4x.

### 2c. Redundant stat() in __contains__
pathlib is_file() 6.66 us; os.path.exists 1.10 us; full torch.load 143.6 us. The stat is 4.6% of a load and
pure waste (getitem reopens). pathlib.is_file is 6x slower than os.path.exists for the same answer.

### 2d. mmap warm reads (50 MiB tensor)
torch.load full 17.85 ms; torch.load(mmap=True)+touch 1 elt 0.147 ms => 121x when you don't need the whole tensor.

## 3. DataLoader num_workers>0 footgun
Memory cacher is a plain in-process dict. Workers are separate processes; whatever they cache never
reaches the parent, and with default persistent_workers=False it's discarded at epoch end.
n=64, 2 ms/miss, cold epoch ~128 ms:

| config | epoch times (ms) | parent cache | later epochs warm? |
|--------|------------------|:------------:|:------------------:|
| num_workers=0                    | 146, 1, 1     | 64/64 | YES |
| num_workers=2, persistent=False  | 776, 104, 112 | 0/64  | NO |
| num_workers=2, persistent=True   | 116, 10, 8    | 0/64  | yes (private) |
| num_workers=4, persistent=True   | 81, 7, 6      | 0/64  | yes (private) |
| num_workers=4, Manager().dict()  | 151, 83, 49   | 64/64 | partial |

1. Default persistent_workers=False -> Memory cache is completely dead under workers; every epoch recomputes.
2. Parent-process cache stays EMPTY (0/64) in every worker config except Manager. Any main-process reuse
   (validation, save, apply/reduce) recomputes from scratch, while the cache *appears* to work.
3. persistent_workers=True warms only each worker's private partial slice; no cross-worker sharing; shuffle degrades hit rate.
4. Fix: Memory(cache=mp.Manager().dict()) restores parent visibility (64/64) but warm epochs 49-83 ms vs 1 ms
   (IPC proxy cost). For real datasets a disk cacher is better - the filesystem is the shared store.

## 4. Prototype optimizations
read_lazy = __getitem__ as-is; read_touch = force full materialization (fair for mmap variants).

| kind | cacher | write sps | read_lazy | read_touch | disk |
|------|--------|----------:|----------:|-----------:|------|
| small  | Pickle(orig)    | 18,545 | 17,532 | -     | 247 KB |
| small  | PickleP5        | 22,202 | 22,623 | -     | 247 KB |
| small  | Tensor(orig p2) | 16,309 | 10,801 | -     | 780 KB |
| small  | TensorP5        | 18,908 | 13,488 | -     | 780 KB |
| small  | MmapTensor      | 19,915 | 12,309 | 4,828 | 780 KB |
| small  | ShardedTensor   | 34,396 | 27,215 | 24,532| ~250 KB |
| medium | Pickle(orig)    | 3,363  | 1,572  | -     | 100 MB |
| medium | PickleP5        | 3,008  | 1,528  | -     | 100 MB |
| medium | Tensor(orig p2) | 2,173  | 4,522  | -     | 100 MB |
| medium | TensorP5        | 2,129  | 4,939  | -     | 100 MB |
| medium | MmapTensor      | 1,977  | 8,048  | 5,107 | 100 MB |
| medium | ShardedTensor   | 1,713  | 4,543  | 4,037 | 100 MB |
| large  | Pickle(orig)    | 6.9    | 42.5   | -     | 500 MB |
| large  | PickleP5        | 17.0   | 42.5   | -     | 500 MB |
| large  | Tensor(orig p2) | 10.0   | 52.4   | -     | 500 MB |
| large  | TensorP5        | 19.8   | 52.3   | -     | 500 MB |
| large  | MmapTensor      | 30.4   | 6,277  | 475.7 | 500 MB |
| large  | ShardedTensor   | 21.8   | 16.6   | 17.3  | 500 MB |

- PickleP5: small write 1.20x, read 1.29x; large write 2.46x (6.9->17.0). Same/smaller disk. Zero-risk.
- TensorP5: small read 1.25x; large write 1.98x (10.0->19.8). One-character change.
- MmapTensor: medium read 1.78x materialized; large materialized 9.1x (52->476), >100x if only sliced.
  Best for large tensors. Caveat: returns mmap-backed tensor (mutate/lifetime footgun) - document.
- ShardedTensor (single blob + offset index + one mmap): standout for MANY SMALL samples - small write 2.11x,
  read 2.3-5x (24,532 vs 10,801 touch), 1 inode instead of N. Poor for large single tensors (slicing the
  mmap copies the whole 50 MB blob before pickle.loads). Right tool for high-count/small-payload datasets.

## 5. Broader avenues
### 5a. Cython on the hot path
cythonize (pure-Python mode, untyped) on apply_mapping/reversed_enumerate/get_sample builds cleanly:
pure-Python get_sample 385.9 ns/call; Cython 311.5 ns/call => 1.24x. Modest; untyped pure-mode can't do
much with dict lookups. Would need cdef typing + .pxd for more, at the cost of a compile step. Not worth it.

### 5b. Per-__getitem__ Python overhead (warm Memory hit, ns/call)
bare dict[idx] 36.4; (idx in d); d[idx] 60.2; d.get(idx) 40.3; td.Dataset.__getitem__ 553.9.
- Wrapper adds ~509 ns/call - 14x a bare dict lookup. For memory-cached small samples this DWARFS the actual
  data access (36 ns). Dominant cost for cache-hit-heavy training, not I/O.
- Double lookup (contains then getitem) 60 vs 40 ns (~1.5x) vs merged get. On disk cachers it's a double FS touch.
- Where 509 ns goes: reversed_enumerate allocs range+reversed+zip every call; apply_mapping slices twice per
  access even with zero maps; extra Python frames (__getitem__ -> get_sample -> apply_mapping) + tuple pack/unpack.

### 5c. Algorithmic wins
- Guard empty-maps: `if start < end` before slicing in apply_mapping; avoid the list copy.
- Replace reversed_enumerate with `for i in range(len(cachers)-1,-1,-1)` - no zip/range/reversed allocation.
- Optional merged fast path: if a cacher exposes get(index, MISS) use one call instead of contains+getitem;
  Memory -> dict.get, disk -> single EAFP `try: open` that also kills the redundant stat.

## Ranked recommendations (biggest I/O win first)
1. Default Tensor (and add proto to Pickle) to pickle_protocol=5. One line. Large-tensor write up to 2.0-2.5x,
   1 MiB read 1.8x, pickle read up to 23x, files 33% smaller. Zero risk, universal. Upstream first.
2. Fix/document the DataLoader Memory-cache footgun. With default workers the dict cache silently does nothing
   and the parent cache is always empty. Warn on Memory + num_workers>0; document Memory(cache=Manager().dict())
   and steer heavy users to disk cachers. Correctness bug in disguise.
3. Ship an MmapTensor cacher for large samples: warm materialized read 9x, lazy/sliced >100x, trivial code
   (torch.load(..., mmap=True)). Document mutate/lifetime caveat.
4. Kill the redundant double-touch: __contains__ -> os.path.exists (6x cheaper than pathlib.is_file) + EAFP
   merged-get fast path so a hit is one op, not stat+open / contains+getitem. Removes ~1.5x + one stat per hit.
5. Trim per-__getitem__ Python overhead (guard empty-map slicing, drop reversed_enumerate allocations). Wrapper
   is 14x a bare dict lookup today; dominant cost for memory-cached small samples where I/O is irrelevant.
6. Offer ShardedTensor (single-file, LMDB-style) for many-small-sample datasets: 2x write, 2-5x read, 1 inode
   instead of N. Not for large single tensors.
7. Skip Cython: builds but only 1.24x untyped; algorithmic wins are larger and free of a compile step.

### Worth upstreaming
protocol-5 default and the DataLoader footgun fix/warning are no-brainers. MmapTensor and ShardedTensor are
worth adding as new named cachers (additive). Lookup-path micro-opts are a small PR. Cython is not worth it.
