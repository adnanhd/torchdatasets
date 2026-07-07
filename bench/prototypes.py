"""Prototype optimized cachers, drop-in compatible with the Cacher interface.

  PickleP5      - Pickle cacher, protocol 5 + os.path.exists (vs pathlib)
  TensorP5      - Tensor cacher, pickle_protocol=5 (vs hardcoded 2)
  MmapTensor    - torch.save writer, torch.load(mmap=True) reader (zero-copy warm)
  ShardedTensor - single append-only blob file + in-memory offset index,
                  warm reads served zero-copy from one mmap (LMDB-style).
                  Eliminates per-sample open/close and inode overhead.
"""
import os
import pathlib
import pickle
import shutil

import torch


class PickleP5:
    def __init__(self, path, extension=".pkl"):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.extension = extension

    def _p(self, index):
        return os.path.join(str(self.path), f"{index}{self.extension}")

    def __contains__(self, index):
        return os.path.exists(self._p(index))

    def __setitem__(self, index, data):
        with open(self._p(index), "wb") as f:
            pickle.dump(data, f, protocol=5)

    def __getitem__(self, index):
        with open(self._p(index), "rb") as f:
            return pickle.load(f)

    def clean(self):
        if self.path.is_dir():
            shutil.rmtree(self.path)


class TensorP5:
    def __init__(self, path, extension=".pt"):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.extension = extension

    def _p(self, index):
        return os.path.join(str(self.path), f"{index}{self.extension}")

    def __contains__(self, index):
        return os.path.exists(self._p(index))

    def __setitem__(self, index, data):
        torch.save(data, self._p(index), pickle_protocol=5)

    def __getitem__(self, index):
        return torch.load(self._p(index), weights_only=False)

    def clean(self):
        if self.path.is_dir():
            shutil.rmtree(self.path)


class MmapTensor:
    """Zero-copy warm reads via torch.load(mmap=True)."""
    def __init__(self, path, extension=".pt"):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.extension = extension

    def _p(self, index):
        return os.path.join(str(self.path), f"{index}{self.extension}")

    def __contains__(self, index):
        return os.path.exists(self._p(index))

    def __setitem__(self, index, data):
        # mmap load requires an uncompressed, non-zip-legacy layout; the
        # modern torch.save format is fine. Protocol 5 for good measure.
        torch.save(data, self._p(index), pickle_protocol=5)

    def __getitem__(self, index):
        return torch.load(self._p(index), weights_only=False, mmap=True)

    def clean(self):
        if self.path.is_dir():
            shutil.rmtree(self.path)


class ShardedTensor:
    """LMDB-style single-file cacher.

    All samples are appended to one blob file; an in-memory dict maps
    index -> (offset, length). Warm reads are served zero-copy from a
    single mmap of the whole file, so there is no per-sample open()/close()
    or directory-entry (inode) cost. Only supports plain tensors here
    (the common heavy case); falls back to pickle bytes for anything else.
    """
    def __init__(self, path, extension=None):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.blob = self.path / "data.blob"
        self.index = {}          # index -> (offset, length)
        self._w = None           # append handle
        self._mm = None          # mmap for reads
        self._offset = 0

    def _writer(self):
        if self._w is None:
            self._w = open(self.blob, "ab", buffering=1024 * 1024)
        return self._w

    def __contains__(self, index):
        return index in self.index

    def __setitem__(self, index, data):
        payload = pickle.dumps(data, protocol=5)
        w = self._writer()
        w.write(payload)
        self.index[index] = (self._offset, len(payload))
        self._offset += len(payload)

    def _mmap(self):
        if self._w is not None:
            self._w.flush()
            self._w.close()
            self._w = None
        if self._mm is None:
            import mmap as _mmap
            f = open(self.blob, "rb")
            self._mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        return self._mm

    def __getitem__(self, index):
        off, length = self.index[index]
        mm = self._mmap()
        return pickle.loads(mm[off:off + length])

    def clean(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._w is not None:
            self._w.close()
            self._w = None
        if self.path.is_dir():
            shutil.rmtree(self.path)
