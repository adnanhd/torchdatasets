:github_url: https://github.com/adnanhd/torchdatasets

*************
torchdatasets
*************

**torchdatasets** is a PyTorch-oriented library focused on data processing and
input pipelines. It extends :class:`torch.utils.data.Dataset` with
functionalities known from ``tensorflow.data`` -- ``map``, ``cache``,
``apply``, ``reduce``, ``filter`` -- with minimal interference (a single call
to ``super().__init__()``) in the original PyTorch datasets.

Overview
########

* Use ``map``, ``apply``, ``reduce`` or ``filter`` directly on ``Dataset`` objects
* ``cache`` data in RAM or on disk, even partially (say the first ``20%``)
* Full PyTorch ``Dataset`` and ``IterableDataset`` support (including
  ``torchvision`` datasets via ``td.datasets.WrapDataset``)
* General ``torchdatasets.maps`` like ``Flatten`` or ``Select``
* Concrete ``torchdatasets.datasets`` designed for file reading and other
  general tasks
* Extensible: bring your own cache methods, cache modifiers, and maps

Installation
############

.. code-block:: shell

  pip install torchdatasets

**torchdatasets** supports Python ``3.7`` to ``3.14`` and PyTorch ``1.8`` to
``2.12``. The core needs only PyTorch. ``torchvision`` is an optional extra used
by the dataset wrappers.

Optional dependency extras:

.. code-block:: shell

  pip install torchdatasets[vision]   # torchvision wrappers (td.datasets.WrapDataset)
  pip install torchdatasets[docs]     # tooling to build these docs
  pip install torchdatasets[dev]      # test / lint / build toolchain

To install the latest, unreleased version straight from ``master``:

.. code-block:: shell

  pip install "git+https://github.com/adnanhd/torchdatasets.git"

Quick start
###########

.. code-block:: python

  import torchdatasets as td
  import torchvision

  class Images(td.Dataset):          # inherit from torchdatasets.Dataset
      def __init__(self, path: str):
          super().__init__()         # the only required call
          self.files = list(path)

      def __getitem__(self, index):
          return Image.open(self.files[index])

      def __len__(self):
          return len(self.files)

  images = (
      Images("./data")
      .map(torchvision.transforms.ToTensor())   # apply a transform
      .cache()                                   # cache in RAM after first pass
  )

Choosing a cacher
#################

``cache`` accepts any :class:`torchdatasets.cachers.Cacher`. Pick one based on
where the samples should live and how many of them there are:

:class:`~torchdatasets.cachers.Memory` (the default)
    Keeps samples in an in-process Python ``dict``. It is the fastest option but
    is bounded by RAM and is gone when the process exits. **Footgun:** under
    ``torch.utils.data.DataLoader(num_workers > 0)`` each worker is a separate
    forked process with its **own** copy of the dict. Samples cached inside a
    worker are never seen by the main process or the other workers. With the
    default ``persistent_workers=False`` every epoch recomputes from scratch. To
    share a cache across workers, pass a manager dict such as
    ``td.cachers.Memory(multiprocessing.Manager().dict())``, which then pays an
    inter-process cost per hit. For most pipelines a disk cacher is the better
    choice.

:class:`~torchdatasets.cachers.Pickle` / :class:`~torchdatasets.cachers.Tensor`
    Persist one file per sample on disk, via ``pickle`` and ``torch.save``
    respectively. The cache survives across separate runs as long as you keep
    the sampling order and seed reproducible.

:class:`~torchdatasets.cachers.MmapTensor`
    Like ``Tensor`` on writes, but warm reads are memory-mapped (zero-copy) via
    ``torch.load(mmap=True)``. Best for large samples: repeat epochs get much
    faster and resident memory stays low, especially when you touch only a slice
    of each sample. Requires ``torch >= 2.1``.

:class:`~torchdatasets.cachers.Sharded`
    A single-file, LMDB-style store. Every sample is appended to one blob, and
    an in-memory offset index plus an ``mmap`` serve warm reads with no
    per-sample file ``open``. Best when the dataset has many small samples. The
    offset index lives in memory, so this cacher reuses data within a single run
    rather than across runs. Populate it on the first epoch and reuse it on the
    rest. Do not share it across ``DataLoader`` workers.

Modules
#######

.. toctree::
   :glob:
   :maxdepth: 1

   packages/*

.. toctree::
   :hidden:

   related

Contributing
############

Issues and pull requests are welcome on
`GitHub <https://github.com/adnanhd/torchdatasets>`__. See the
`Roadmap <https://github.com/adnanhd/torchdatasets/blob/master/ROADMAP.md>`__
for an overview of planned work.
