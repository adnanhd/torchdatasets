################################################################################
#
#                                HELPERS
#
################################################################################

from __future__ import annotations

import typing

_T = typing.TypeVar("_T")


def apply_mapping(
    sample: typing.Any,
    mappings: typing.Sequence[typing.Callable[[typing.Any], typing.Any]],
    start: int,
    end: int,
) -> typing.Any:
    """Helper applying maps in defined threshold.

    Indexes ``mappings`` in place instead of slicing it, and short-circuits the
    empty range. This runs on every cached ``__getitem__`` (often with
    ``start == end``), so avoiding the per-call list copy matters.
    """
    if start >= end:
        return sample
    for index in range(start, end):
        sample = mappings[index](sample)
    return sample


def reversed_enumerate(
    iterable: typing.Sequence[_T],
) -> typing.Iterator[typing.Tuple[int, _T]]:
    """Yield ``(index, item)`` pairs from last to first.

    A hand-rolled generator is about twice as fast here as
    ``zip(range(...), reversed(...))`` for the tiny cacher lists this walks on
    every ``__getitem__``.
    """
    for index in range(len(iterable) - 1, -1, -1):
        yield index, iterable[index]
