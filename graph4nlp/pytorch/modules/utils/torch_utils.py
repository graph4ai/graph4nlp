import torch


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x
