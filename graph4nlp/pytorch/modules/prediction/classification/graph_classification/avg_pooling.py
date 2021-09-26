import torch

from .....data.data import from_batch
from ..base import PoolingBase


class AvgPooling(PoolingBase):
    r"""Apply average pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
    """

    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute average pooling.

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
            output_feat.append(g.node_features[feat].mean(dim=0))

        output_feat = torch.stack(output_feat, 0)

        return output_feat
