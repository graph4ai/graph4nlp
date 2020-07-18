import torch.nn as nn

class GNNLayerBase(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayerBase, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        pass

    def forward(self, graph, node_feats):
        raise NotImplementedError('GNNLayerBase: Not Implemented.')

class GNNBase(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNBase, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        pass

    def forward(self, graph, node_feats):
        raise NotImplementedError('GNNBase: Not Implemented.')