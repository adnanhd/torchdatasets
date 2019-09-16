r"""**This module allows you to modify behaviour of** `torchdata.cachers`.

To cache in `memory` only `20` first samples you could do (assuming you have already created
`torchdata.Dataset` instance named `dataset`)::

    dataset.cache(torchdata.modifiers.UpToIndex(torchdata.cachers.Memory(), 20))

Modifers could be mixed intuitvely as well using logical operators `|` (or) and
`&` (and).

**Example** (cache to disk `20` first or samples with index `1000` and upwards)::

    dataset.cache(
        torchdata.modifiers.UpToIndex(torchdata.cachers.Memory(), 20)
        | torchdata.modifiers.FromIndex(torchdata.cachers.Memory(), 1000)
    )

You can mix provided modifiers or extend them by inheriting from `Modifier`
and implementing `condition` method (interface described below).

"""

import abc

from ._base import Base


class Modifier(Base):
    r"""**Interface for all modifiers.**

    Most methods are pre-configured, so user should not override them.
    In-fact only `condition` has to be overriden and `__init__` implemented.
    Constructor should assign `cacher` to `self` in order for everything
    to work, see example below.

    Example implementation of `modifier` caching only elements `0` to `100`
    of any `torchdata.cacher.Cacher`::

        class ExampleModifier(Modifier):

            # You have to assign cacher to self.cacher so modifier works.
            def __init__(self, cacher):
                self.cacher = cacher

            def condition(self, index):
                return index < 100 # Cache if index smaller than 100

    """

    @abc.abstractmethod
    def condition(self, index) -> bool:
        r"""**Based on index, decide whether cache should interact with the sample.**

        Only this function should be implemented by user.
        If `True` returned, `cacher` will act on sample normally (e.g. saving it or loading).

        Parameters
        ----------
        index : int
                Index of sample

        Returns
        -------
        bool
                Whether to act on sample with given index

        """
        pass

    def __contains__(self, index: int) -> bool:
        r"""**Acts as invisible proxy for** `cacher`'s `__contains__` **method.**

        **User should not override this method.**
        For more information check `torchdata.cacher.Cacher` interface.

        Parameters
        ----------
        index : int
                Index of sample

        """
        if self.condition(index):
            return index in self.cacher
        return False

    def __setitem__(self, index: int, data: int) -> None:
        r"""**Acts as invisible proxy for** `cacher`'s `__setitem__` **method.**

        **User should not override this method.**
        For more information check `torchdata.cacher.Cacher` interface.

        Parameters
        ----------
        index : int
                Index of sample
        data : Any
                Data generated by dataset.
        """
        if self.condition(index):
            self.cacher[index] = data

    def __getitem__(self, index: int):
        r"""**Acts as invisible proxy for** `cacher`'s `__getitem__` **method.**

        **User should not override this method.**
        For more information check `torchdata.cacher.Cacher` interface.

        Parameters
        ----------
        index : int
                Index of sample
        """
        return self.cacher[index]

    def __or__(self, other):
        r"""**If self or other returns True, then use** `cacher`.

        User should not override this method.

        **Important:** `self` and `other` should have the same `cacher` wrapped.
        Otherwise exception is thrown. Cacher of first modifier is used in such case.

        Parameters
        ----------
        other : Modifier
                Another modifier

        Returns
        -------
        torchdata.modifiers.Any
                Modifier concatenating both modifiers.

        """
        return Any(self, other)

    def __and__(self, other):
        r"""**If self and other returns True, then use** `cacher`.

        **Important:** `self` and `other` should have the same `cacher` wrapped.
        Otherwise exception is thrown. Cacher of first modifier is used in such case.

        Parameters
        ----------
        other : Modifier
                Another modifier

        Returns
        -------
        torchdata.modifiers.All
                Modifier concatenating both modifiers.

        """
        return All(self, other)


class _Mix(Modifier):
    r"""**{}**

    Parameters
    ----------
    *modifiers: List[torchdata.modifiers.Modifier]
            List of modifiers

    """

    def __init__(self, *modifiers):
        first_type = type(modifiers[0])
        self.modifiers = modifiers
        if not all(isinstance(checker, first_type) for checker in self.modifiers):
            raise ValueError("All checkers have to be of same type")

        self.cacher = modifiers[0].cacher


class All(_Mix):
    __doc__ = _Mix.__doc__.format(
        r"Return True if all modifiers return True on given sample."
    )

    def condition(self, index):
        return all(modifier.condition() for modifier in self.modifiers)


class Any(_Mix):
    __doc__ = _Mix.__doc__.format(
        r"Return True if any modifier returns True on given sample."
    )

    def condition(self, index):
        return any(modifier.condition() for modifier in self.modifiers)


class _Percent(Modifier):
    r"""**{}**

    Parameters
    ----------
    cacher : torchdata.cacher.Cacher
            Instance of cacher
    p : float
            Percentage specified as flow between `[0, 1]`.
    length : int
            How many samples are in dataset. You can pass `len(dataset)`.

    """

    @abc.abstractmethod
    def condition(self, index):
        pass

    def __init__(self, cacher, p: float, length: int):
        if not 0 < p < 1:
            raise ValueError(f"Percentage has to be between 0 and 1, but got {p}")
        self.threshold: int = int(length * p)
        self.cacher = cacher


class UpToPercentage(_Percent):
    __doc__ = _Percent.__doc__.format(
        r"""Cache up to percentage of samples leaving the rest untouched."""
    )

    def condition(self, index):
        return index < self.threshold


class FromPercentage(_Percent):
    __doc__ = _Percent.__doc__.format(
        r"""Cache from specified percentage of samples leaving the rest untouched."""
    )

    def condition(self, index):
        return index > self.threshold


class _Index(Modifier):
    r"""**{}**

    Parameters
    ----------
    cacher : torchdata.cacher.Cacher
            Instance of cacher
    index : int
            Index of sample

    """

    @abc.abstractmethod
    def condition(self, index):
        pass

    def __init__(self, cacher, index: int):
        self.cacher = cacher
        self.index: int = index


class UpToIndex(_Index):
    __doc__ = _Index.__doc__.format(
        r"""Cache up to samples of specified index leaving the rest untouched."""
    )

    def condition(self, index):
        return index < self.index


class FromIndex(_Index):
    __doc__ = _Index.__doc__.format(
        r"""Cache samples from specified index leaving the rest untouched."""
    )

    def condition(self, index):
        return index > self.index


class Indices(Modifier):
    r"""**Cache samples if index is one of specified.**

    Parameters
    ----------
    cacher : List[torchdata.modifiers.Modifier]
            List of modifiers
    index : int
            Index of sample

    """

    def __init__(self, cacher, *indices):
        self.cacher = cacher
        self.indices = indices

    def condition(self, index):
        return index in self.indices
