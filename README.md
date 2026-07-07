## Package renamed to torchdatasets!

<img align="left" width="256" height="256" src="https://github.com/adnanhd/torchdatasets/blob/master/assets/logos/medium.png">

* Use `map`, `apply`, `reduce` or `filter` directly on `Dataset` objects
* `cache` data in RAM/disk or via your own method (partial caching supported)
* Full PyTorch's [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>) support
* General `torchdatasets.maps` like `Flatten` or `Select`
* Extensible interface (your own cache methods, cache modifiers, maps etc.)
* Useful `torchdatasets.datasets` classes designed for general tasks (e.g. file reading)
* Support for `torchvision` datasets (e.g. `ImageFolder`, `MNIST`, `CIFAR10`) via `td.datasets.WrapDataset`
* Minimal overhead (single call to `super().__init__()`)

| Version | Docs | Tests | Coverage | PyPI | Python | PyTorch | Roadmap |
|---------|------|-------|----------|------|--------|---------|---------|
| [![Version](https://img.shields.io/static/v1?label=&message=0.2.0&color=377EF0&style=for-the-badge)](https://github.com/adnanhd/torchdatasets/releases) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](https://adnanhd.github.io/torchdatasets/) | [![Tests](https://github.com/adnanhd/torchdatasets/actions/workflows/ci.yml/badge.svg)](https://github.com/adnanhd/torchdatasets/actions/workflows/ci.yml) | ![Coverage](https://img.shields.io/codecov/c/github/adnanhd/torchdatasets?label=%20&logo=codecov&style=for-the-badge) | [![PyPI](https://img.shields.io/static/v1?label=&message=PyPI&color=377EF0&style=for-the-badge)](https://pypi.org/project/torchdatasets/) | [![Python](https://img.shields.io/static/v1?label=&message=3.7%20|%203.14&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=>=1.8&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Roadmap](https://img.shields.io/static/v1?label=&message=roadmap&color=009688&style=for-the-badge)](https://github.com/adnanhd/torchdatasets/blob/master/ROADMAP.md) |

# :bulb: Examples

__Check documentation here:__
[https://adnanhd.github.io/torchdatasets](https://adnanhd.github.io/torchdatasets)

## General example

- Create image dataset, convert it to Tensors, cache and concatenate with smoothed labels:

```python
import torchdatasets as td
import torchvision

class Images(td.Dataset): # Different inheritance
    def __init__(self, path: str):
        super().__init__() # This is the only change
        self.files = [file for file in pathlib.Path(path).glob("*")]

    def __getitem__(self, index):
        return Image.open(self.files[index])

    def __len__(self):
        return len(self.files)


images = Images("./data").map(torchvision.transforms.ToTensor()).cache()
```

You can concatenate above dataset with another (say `labels`) and iterate over them as per usual:

```python
for data, label in images | labels:
    # Do whatever you want with your data
```

- Cache first `1000` samples in memory, save the rest on disk in folder `./cache`:

```python
images = (
    ImageDataset.from_folder("./data").map(torchvision.transforms.ToTensor())
    # First 1000 samples in memory
    .cache(td.modifiers.UpToIndex(1000, td.cachers.Memory()))
    # Sample from 1000 to the end saved with Pickle on disk
    .cache(td.modifiers.FromIndex(1000, td.cachers.Pickle("./cache")))
    # You can define your own cachers, modifiers, see docs
)
```
To see what else you can do please check [**torchdatasets documentation**](https://adnanhd.github.io/torchdatasets/)

## Integration with `torchvision`

Using `torchdatasets` you can easily split `torchvision` datasets and apply augmentation
only to the training part of data without any troubles:

```python
import torchvision

import torchdatasets as td

# Wrap torchvision dataset with WrapDataset
dataset = td.datasets.WrapDataset(torchvision.datasets.ImageFolder("./images"))

# Split dataset
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    model_dataset,
    (int(0.6 * len(dataset)), int(0.2 * len(dataset)), int(0.2 * len(dataset))),
)

# Apply torchvision mappings ONLY to train dataset
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
    # Apply this transformation to zeroth sample
    # First sample is the label
    0,
)
```

Please notice you can use `td.datasets.WrapDataset` with any existing `torch.utils.data.Dataset`
instance to give it additional `caching` and `mapping` powers!

# :wrench: Installation

## :snake: [pip](<https://pypi.org/project/torchdatasets/>)

### Latest release:

```shell
pip install --user torchdatasets
```

### From source:

```shell
pip install --user "git+https://github.com/adnanhd/torchdatasets.git"
```

# :question: Contributing

If you find any issue or you think some functionality may be useful to others and fits this library, please [open new Issue](https://help.github.com/en/articles/creating-an-issue) or [create Pull Request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

To get an overview of thins one can do to help this project, see [Roadmap](https://github.com/adnanhd/torchdatasets/blob/master/ROADMAP.md)
