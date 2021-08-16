import torch.nn as nn


class GeneralLossBase(nn.Module):
    def __init__(self):
        super(GeneralLossBase, self).__init__()

    def forward(self, logits, label, **kwargs):
        raise NotImplementedError("GeneralLossBase: Not Implemented.")


class KGLossBase(nn.Module):
    def __init__(self):
        super(KGLossBase, self).__init__()

    def forward(self, graph, node_feat):
        raise NotImplementedError("KGLossBase: Not Implemented.")
