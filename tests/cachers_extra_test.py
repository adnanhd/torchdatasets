"""Dependency-free cacher tests.

Unlike ``cachers_test.py`` (which needs ``torchfunc``/``torchvision`` and is
skipped on the CI version matrix), these exercise the on-disk cache mechanics
with only ``torch`` + ``torchdatasets`` so they run on every torch/python row.
"""

import pickle

import pytest
import torch

import torchdatasets as td
from torchdatasets.cachers import _torch_version


def test_pickle_roundtrip_and_protocol(tmp_path):
    with td.cachers.Pickle(tmp_path / "pkl") as cacher:
        payload = {"a": torch.arange(4), "b": [1, 2, 3]}
        assert 0 not in cacher
        cacher[0] = payload
        assert 0 in cacher
        out = cacher[0]
    assert torch.equal(out["a"], payload["a"])
    assert out["b"] == payload["b"]


def test_tensor_roundtrip(tmp_path):
    with td.cachers.Tensor(tmp_path / "pt") as cacher:
        t = torch.randn(8, 8)
        cacher[3] = t
        assert 3 in cacher
        assert torch.equal(cacher[3], t)


def test_tensor_caches_non_tensor_objects(tmp_path):
    """Regression guard: torch 2.6 flipped ``weights_only`` to True, which would
    refuse to reload an arbitrary (non-tensor) cached sample."""
    with td.cachers.Tensor(tmp_path / "pt-obj") as cacher:
        payload = {"meta": "label", "nums": [1, 2, 3], "t": torch.ones(2)}
        cacher[0] = payload
        out = cacher[0]
    assert out["meta"] == "label"
    assert out["nums"] == [1, 2, 3]
    assert torch.equal(out["t"], torch.ones(2))


def test_tensor_default_protocol_is_highest():
    # Default constructor protocol should be the best the interpreter offers.
    import inspect

    default = (
        inspect.signature(td.cachers.Tensor.__init__)
        .parameters["pickle_protocol"]
        .default
    )
    assert default == pickle.HIGHEST_PROTOCOL


def test_sharded_within_run_reuse(tmp_path):
    with td.cachers.Sharded(tmp_path / "shard") as cacher:
        samples = {i: {"i": i, "t": torch.full((3,), float(i))} for i in range(5)}
        for i, s in samples.items():
            assert i not in cacher
            cacher[i] = s
        # read back (mmap path)
        for i, s in samples.items():
            assert i in cacher
            out = cacher[i]
            assert out["i"] == i
            assert torch.equal(out["t"], s["t"])


def test_sharded_interleaved_write_read(tmp_path):
    """Writing after a read must invalidate the stale mmap and still read back."""
    with td.cachers.Sharded(tmp_path / "shard2") as cacher:
        cacher[0] = torch.tensor([0.0])
        assert torch.equal(cacher[0], torch.tensor([0.0]))  # builds mmap
        cacher[1] = torch.tensor([1.0])  # grows file, invalidates mmap
        assert torch.equal(cacher[1], torch.tensor([1.0]))
        assert torch.equal(cacher[0], torch.tensor([0.0]))  # old offset still valid


def test_mmap_tensor_roundtrip_or_guarded(tmp_path):
    if _torch_version() < (2, 1):
        with pytest.raises(RuntimeError):
            td.cachers.MmapTensor(tmp_path / "mmap")
        return
    with td.cachers.MmapTensor(tmp_path / "mmap") as cacher:
        t = torch.randn(16, 16)
        cacher[7] = t
        assert 7 in cacher
        assert torch.equal(cacher[7], t)


def test_sharded_discards_stale_blob(tmp_path):
    """A leftover blob from a prior run must not corrupt a fresh instance."""
    d = tmp_path / "shard_stale"
    d.mkdir()
    (d / "data.blob").write_bytes(b"garbage bytes from a previous, uncleaned run")
    with td.cachers.Sharded(d) as cacher:
        cacher[0] = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(cacher[0], torch.tensor([1.0, 2.0, 3.0]))


def test_pickle_protocol_override(tmp_path):
    """A lower protocol can be requested for cross-Python cache sharing."""
    with td.cachers.Pickle(tmp_path / "pkl4", protocol=4) as cacher:
        cacher[0] = {"x": torch.arange(3)}
        assert torch.equal(cacher[0]["x"], torch.arange(3))


def test_cachers_accept_str_paths(tmp_path):
    """Path arguments should accept plain strings too."""
    with td.cachers.Pickle(str(tmp_path / "p")) as c:
        c[0] = 1
        assert c[0] == 1
    with td.cachers.Tensor(str(tmp_path / "t")) as c:
        c[0] = torch.zeros(2)
        assert torch.equal(c[0], torch.zeros(2))
    with td.cachers.Sharded(str(tmp_path / "s")) as c:
        c[0] = torch.ones(2)
        assert torch.equal(c[0], torch.ones(2))


def test_cache_flow_through_dataset(tmp_path):
    """End-to-end: a Dataset with a disk cacher recomputes once then reuses."""
    calls = {"n": 0}

    class Counting(td.Dataset):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def __getitem__(self, index):
            calls["n"] += 1
            return torch.tensor([float(index)])

        def __len__(self):
            return self.n

    with td.cachers.Tensor(tmp_path / "flow") as cacher:
        dataset = Counting(4).cache(cacher)
        first = [dataset[i] for i in range(4)]
        second = [dataset[i] for i in range(4)]
    assert calls["n"] == 4  # second pass served from cache
    for a, b in zip(first, second):
        assert torch.equal(a, b)
