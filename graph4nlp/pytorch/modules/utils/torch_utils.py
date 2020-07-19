import torch


def to_cuda(x, use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        x = x.cuda()
    return x
