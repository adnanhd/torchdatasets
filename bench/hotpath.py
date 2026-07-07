"""Hot-path functions copied from torchdatasets internals, for Cython.

These mirror _dev_utils.apply_mapping / reversed_enumerate and the
_base.get_sample cache-lookup loop. Cython can compile this .py directly
(pure-python mode) with `cythonize`.
"""


def apply_mapping(sample, mappings, start, end):
    for mapping in mappings[start:end]:
        sample = mapping(sample)
    return sample


def reversed_enumerate(iterable):
    return zip(range(len(iterable) - 1, -1, -1), reversed(iterable))


def get_sample(cachers, which, maps, index, original):
    """Cache-lookup path mirroring _base.MetaDataset.get_sample."""
    for cacher_index, cacher in reversed_enumerate(cachers):
        if index in cacher:
            return cacher[index], which[cacher_index]
    sample = original(index)
    most_mappings = 0
    for cacher_index, cacher in enumerate(cachers):
        maps_index = which[cacher_index]
        sample = apply_mapping(sample, maps, most_mappings, maps_index)
        most_mappings = maps_index
        cacher[index] = sample
    return sample, most_mappings
