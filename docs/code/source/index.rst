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

The core needs only PyTorch. ``torchvision`` is optional and used by the
dataset wrappers.

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
