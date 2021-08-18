from typing import Any, List, Tuple, Union
import torch


class SizeMismatchException(Exception):
    pass


class NodeNotFoundException(Exception):
    pass


class EdgeNotFoundException(Exception):
    pass


def int_to_list(x: Union[int, List[Any]]):
    if not (isinstance(x, list) or isinstance(x, int)):
        raise TypeError("Input x should be int or list. Got {} instead.".format(type(x)))
    # assert isinstance(x, list) or isinstance(x, int)
    return x if isinstance(x, list) else [x]


def check_and_expand(x: list, y: list) -> Tuple[List[Any], List[Any]]:
    if not (isinstance(x, list) and isinstance(y, list)):
        raise TypeError(
            "Input x and y should be lists. Got {} and {} instead.".format(type(x), type(y))
        )
    # assert isinstance(x, list) and isinstance(y, list)
    max_len = max(len(x), len(y))
    if len(x) == len(y):
        return x, y
    elif len(x) * len(y) == max_len:  # Which means at least one of the list is of length 1
        if len(x) == 1:
            x = x * max_len
        else:
            y = y * max_len
        return x, y
    else:
        raise ValueError(
            "The two lists {} and {} cannot be automatically "
            "broadcasted to the same length.".format(x, y)
        )


def slice_to_list(sl: slice, max_len: int) -> List[int]:
    """
    Turn a slice object into a list

    Parameters
    ----------
    sl: slice
        The slice object

    max_len: int
        Max length of the iterable

    Returns
    -------
    list
        The converted list
    """

    if sl.start is None:
        start = 0
    elif sl.start < 0:
        start = max_len + sl.start
    else:
        start = sl.start

    if sl.stop is None:
        stop = max_len
    elif sl.stop < 0:
        stop = max_len + sl.stop
    else:
        stop = sl.stop

    if sl.step is None:
        step = 1
    else:
        step = sl.step

    return list(range(start, stop, step))


def entail_zero_padding(old_tensor: torch.Tensor, num_rows: int) -> torch.Tensor:
    if old_tensor is None:
        return None

    if len(old_tensor.shape) == 1:
        return torch.cat(
            (old_tensor, torch.zeros(num_rows).to(dtype=old_tensor.dtype, device=old_tensor.device))
        )
    else:
        return torch.cat(
            (
                old_tensor,
                torch.zeros((num_rows, *old_tensor.shape[1:])).to(
                    dtype=old_tensor.dtype, device=old_tensor.device
                ),
            ),
            dim=0,
        )


def reverse_index(l: List[Any], v: Any):
    """
    Find the index of the last occurrence of an element in a list.

    Parameters
    ----------
    l: list
        The container of all elements
    v: object
        The element to be found

    Returns
    -------
    int:
        The index of the last occurrence of `v` in `l`.

    Raises
    ------
    ValueError
        If the element is not found in the list.
    """
    if v not in l:
        raise ValueError("Given value v not found in the list")
    return len(l) - l[::-1].index(v) - 1
