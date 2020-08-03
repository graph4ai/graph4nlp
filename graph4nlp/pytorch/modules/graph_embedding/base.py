import abc
import torch.nn as nn


class GNNLayerBase(nn.Module):
    def __init__(self):
        super(GNNLayerBase, self).__init__()

    @abc.abstractmethod
    def forward(self, graph, node_feat):
        raise NotImplementedError('GNNLayerBase: Not Implemented.')

class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    @abc.abstractmethod
    def forward(self, graph, node_feat):
        raise NotImplementedError('GNNBase: Not Implemented.')
