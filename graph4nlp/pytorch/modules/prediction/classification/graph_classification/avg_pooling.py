from torch import nn
from dgl.nn.pytorch.glob import AvgPooling as DGLAvgPooling

from ..base import PoolingBase


class AvgPooling(PoolingBase):
    r"""Apply average pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
    """
    def __init__(self):
        super(AvgPooling, self).__init__()
        self.model = DGLAvgPooling()

    def forward(self, graph, feat):
        r"""Compute average pooling.

        Parameters
        ----------
        graph : GraphData
            The graph data.
        feat : torch.Tensor
            The input feature.

        Returns
        -------
        torch.Tensor
            The output feature.
        """
        # graph = graph.to_dgl()

        return self.model(graph, feat)
