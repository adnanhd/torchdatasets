# torchdatasets

**PyTorch datasets with `map`, `cache`, `apply`, `reduce` and `filter` built in.**

<p>
  <a href="https://github.com/adnanhd/torchdatasets/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/adnanhd/torchdatasets/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/torchdatasets/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchdatasets?style=flat-square&color=377EF0"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.7%20to%203.14-377EF0?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-1.8%20to%202.12-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://adnanhd.github.io/torchdatasets/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-3776AB?style=flat-square"></a>
  <a href="https://github.com/adnanhd/torchdatasets/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
</p>

<img align="left" width="220" height="220" src="https://raw.githubusercontent.com/adnanhd/torchdatasets/master/assets/logos/medium.png" alt="torchdatasets logo">

`torchdatasets` extends `torch.utils.data.Dataset` with the pipeline operations
familiar from `tensorflow.data`, adding only a single `super().__init__()` call
to your dataset.

- Use `map`, `apply`, `reduce` and `filter` directly on `Dataset` objects.
- `cache` data in RAM or on disk, even partially, or through your own cacher.
- Full PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) support.
- General maps such as `Flatten` and `Select` in `torchdatasets.maps`.
- Ready-made datasets for common tasks like file reading.
- Wrap any `torchvision` dataset with `td.datasets.WrapDataset`.
- Extensible: write your own cachers, cache modifiers and maps.

<br clear="left">

## Installation

Latest release from PyPI:

```shell
pip install torchdatasets
```

Latest development version from source:

```shell
pip install "git+https://github.com/adnanhd/torchdatasets.git"
```

`torchdatasets` supports Python 3.7 to 3.14 and PyTorch 1.8 to 2.12. The core
depends only on PyTorch. Install the `vision` extra for the `torchvision`
wrappers:

```shell
pip install "torchdatasets[vision]"
```

## Quick start

Build an image dataset, convert samples to tensors and cache them after the
first pass:

```python
import pathlib

import torchvision
from PIL import Image

import torchdatasets as td


class Images(td.Dataset):          # inherit from torchdatasets.Dataset
    def __init__(self, path: str):
        super().__init__()         # the only required change
        self.files = list(pathlib.Path(path).glob("*"))

    def __getitem__(self, index):
        return Image.open(self.files[index])

    def __len__(self):
        return len(self.files)


images = Images("./data").map(torchvision.transforms.ToTensor()).cache()
```

Concatenate two datasets with `|` and iterate as usual:

```python
for data, label in images | labels:
    ...
```

Cache the first `1000` samples in RAM and the rest on disk:

```python
images = (
    Images.from_folder("./data")
    .map(torchvision.transforms.ToTensor())
    .cache(td.modifiers.UpToIndex(1000, td.cachers.Memory()))
    .cache(td.modifiers.FromIndex(1000, td.cachers.Pickle("./cache")))
)
```

## Working with `torchvision`

Split a `torchvision` dataset and apply augmentation to the training part only:

```python
import torch
import torchvision

import torchdatasets as td

dataset = td.datasets.WrapDataset(torchvision.datasets.ImageFolder("./images"))

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    (int(0.6 * len(dataset)), int(0.2 * len(dataset)), int(0.2 * len(dataset))),
)

train_dataset.map(
    td.maps.To(
        torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    ),
    0,  # apply only to the sample, not the label
)
```

`td.datasets.WrapDataset` works with any existing `torch.utils.data.Dataset`,
giving it caching and mapping powers.

## Documentation

Full documentation, including the cacher selection guide, lives at
[adnanhd.github.io/torchdatasets](https://adnanhd.github.io/torchdatasets/).

## Contributing

Found a bug or have an idea that fits the library? Please
[open an issue](https://github.com/adnanhd/torchdatasets/issues) or send a pull
request. See the [roadmap](https://github.com/adnanhd/torchdatasets/blob/master/ROADMAP.md)
for planned work.
