import warnings
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.nn.pytorch import RelGraphConv

from .base import GNNBase, GNNLayerBase


class RGCN(GNNBase):
    r"""Multi-layered `GraphSAGE Network <https://arxiv.org/pdf/1706.02216.pdf>`__
    Support both unidirectional (i.e., regular) and bidirectional (i.e., `bi_sep` and `bi_fuse`)
    versions.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)
        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    num_layers: int
        Number of GraphSAGE layers.
    input_size : int, or pair of ints
        Input feature size.
        If the layer is to be applied to a unidirectional bipartite graph, ``input_size``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    hidden_size: int list of int
        Hidden layer size.
        If a scalar is given, the sizes of all the hidden layers are the same.
        If a list of scalar is given, each element in the list is the size of each hidden layer.
        Example: [100,50]
    output_size : int
        Output feature size. For the bisep version, the output size is 2*output_size
    num_rels : int
        Number of relations. .
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    use_self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    """

    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        num_rels,
        num_bases=None,
        use_self_loop=True,
        dropout=0.0
    ):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.use_self_loop = use_self_loop
        self.dropout = dropout
        self.use_cuda = use_cuda

        self.RGCN_layers = nn.ModuleList()

        # transform the hidden size format
        if self.num_layers > 1 and type(hidden_size) is int:
            hidden_size = [hidden_size for i in range(self.num_layers - 1)]

        if self.num_layers > 1:
            # input projection
            self.RGCN_layers.append(
                RGCNLayer(
                    input_size,
                    hidden_size[0],
                    num_rels=self.num_rels,
                    regularizer="basis",
                    num_bases=self.num_bases,
                    bias=True,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout
                )
            )
        # hidden layers
        for l in range(1, self.num_layers - 1):
            # due to multi-head, the input_size = hidden_size * num_heads
            self.RGCN_layers.append(
                RGCNLayer(
                    hidden_size[l-1],
                    hidden_size[l],
                    num_rels=self.num_rels,
                    regularizer="basis",
                    num_bases=self.num_bases,
                    bias=True,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout
                )
            )
        # output projection
        self.RGCN_layers.append(
            RGCNLayer(
                hidden_size[-1] if self.num_layers > 1 else input_size,
                output_size,
                num_rels=self.num_rels,
                regularizer="basis",
                num_bases=self.num_bases,
                bias=True,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=self.dropout
            )
        )

    def forward(self, graph):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : GraphData
            The graph with node feature stored in the feature field named as
            "node_feat".
            The node features are used for message passing.

        Returns
        -------
        graph : GraphData
            The graph with generated node embedding stored in the feature field
            named as "node_emb".
        """

        h = graph.node_features["node_feat"]
        # get the node feature tensor from graph
        g = graph.to_dgl()  # transfer the current NLPgraph to DGL graph
        edge_type = g.edata[dgl.ETYPE].long()
        # output projection
        if self.num_layers > 1:
            for l in range(0, self.num_layers - 1):
                h = self.RGCN_layers[l](g, h, edge_type)

        logits = self.RGCN_layers[-1](g, h, edge_type)

        graph.node_features["node_emb"] = logits  # put the results into the NLPGraph
        return graph


class RGCNLayer(GNNLayerBase):
    r"""A wrapper for RelGraphConv in DGL.

    .. math::
        tbd

    Parameters
    ----------
    input_size : int, or pair of ints
        Input feature size.
        If the layer is to be applied to a unidirectional bipartite graph, ``input_size``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    output_size : int
        Output feature size.
    num_rels: int
        number of relations
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":
         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.
        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_rels,
        regularizer=None,
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
        layer_norm=False
    ):
        super(RGCNLayer, self).__init__()
        self.model = RelGraphConv(
                        in_feat=input_size, 
                        out_feat=output_size, 
                        num_rels=num_rels, 
                        regularizer=regularizer,
                        num_bases=num_bases,
                        bias=bias, 
                        activation=activation,
                        self_loop=self_loop,
                        dropout=dropout,
                        layer_norm=layer_norm)

    def forward(self, graph, feat, etypes, norm=None):
        return self.model(graph, feat, etypes, norm)
