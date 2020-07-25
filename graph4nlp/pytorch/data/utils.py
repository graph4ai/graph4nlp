import torch


class SizeMismatchException(Exception):
    pass


class NodeNotFoundException(Exception):
    pass

class EdgeNotFoundException(Exception):
    pass


def slice_to_list(sl, max_len):
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


def entail_zero_padding(old_tensor: torch.tensor, num_rows: int):
    if old_tensor is None:
        return

    if len(old_tensor.shape) == 1:
        torch.cat((old_tensor, torch.zeros(1)), out=old_tensor)
    else:
        torch.cat((old_tensor, torch.zeros((num_rows, *old_tensor.shape[1:]))), dim=0, out=old_tensor)
