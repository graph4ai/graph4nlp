import torch
import torch.nn as nn

from .....data.data import from_batch
from ..base import PoolingBase


class MaxPooling(PoolingBase):
    r"""Apply max pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i}\left( x^{(i)}_k \right)
    """

    def __init__(self, dim=None, use_linear_proj=False):
        super(MaxPooling, self).__init__()
        if use_linear_proj:
            assert dim is not None, "dim should be specified when use_linear_proj is set to True"
            self.linear = nn.Linear(dim, dim, bias=False)
        else:
            self.linear = None

    def forward(self, graph, feat):
        r"""Compute max pooling.

        Parameters
        ----------
        graph : GraphData
            The graph data.
        feat : str
            The feature field name.

        Returns
        -------
        torch.Tensor
            The output feature.
        """
        graph_list = from_batch(graph)
        output_feat = []
        for g in graph_list:
            feat_tensor = g.node_features[feat]
            if self.linear is not None:
                feat_tensor = self.linear(feat_tensor)

            output_feat.append(torch.max(feat_tensor, dim=0)[0])

        output_feat = torch.stack(output_feat, 0)

        return output_feat
