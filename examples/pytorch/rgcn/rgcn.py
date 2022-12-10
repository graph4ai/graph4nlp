import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph4nlp.pytorch.modules.graph_embedding_learning.base import GNNBase, GNNLayerBase


class RGCN(GNNBase):
    r"""Multi-layered `RGCN Network <TODO:paper.pdf>`__

    .. math::
        TODO:Add Calculation.

    Parameters
    ----------
    num_layers: int
        Number of RGCN layers.
    input_size : int, or pair of ints
        Input feature size.
    hidden_size: int list of int
        Hidden layer size.
        If a scalar is given, the sizes of all the hidden layers are the same.
        If a list of scalar is given, each element in the list is the size of each hidden layer.
        Example: [100,50]
    output_size : int
        Output feature size.
    num_rels : int
        Number of relations.
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
        dropout=0.0,
    ):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.use_self_loop = use_self_loop
        self.dropout = dropout

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
                    dropout=self.dropout,
                )
            )
        # hidden layers
        for l in range(1, self.num_layers - 1):
            # due to multi-head, the input_size = hidden_size * num_heads
            self.RGCN_layers.append(
                RGCNLayer(
                    hidden_size[l - 1],
                    hidden_size[l],
                    num_rels=self.num_rels,
                    regularizer="basis",
                    num_bases=self.num_bases,
                    bias=True,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
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
                dropout=self.dropout,
            )
        )

    def forward(self, graph):
        r"""Compute RGCN layer.

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
        # edge_type = g.edata[dgl.ETYPE].long()
        # output projection
        if self.num_layers > 1:
            for l in range(0, self.num_layers - 1):
                h = self.RGCN_layers[l](g, h)

        logits = self.RGCN_layers[-1](g, h)

        graph.node_features["node_emb"] = logits  # put the results into the NLPGraph
        return graph


class RGCNLayer(GNNLayerBase):
    r"""A wrapper for RelGraphConv in DGL.

    .. math::
        TODO

    Parameters
    ----------
    input_size : int, or pair of ints
        Input feature size.
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
        layer_norm=False,
    ):
        super(RGCNLayer, self).__init__()
        self.linear_dict = {
            i: nn.Linear(input_size, output_size, bias=bias) for i in range(num_rels)
        }
        # self.linear_r = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(output_size))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(output_size, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(input_size, output_size))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLHeteroGraph, feat: torch.Tensor, norm=None):
        def message(edges, g):
            """Message function."""
            ln = self.linear_dict[g.canonical_etypes.index(edges._etype)]
            m = ln(edges.src["h"])
            if "norm" in edges.data:
                m = m * edges.data["norm"]
            return {"m": m}

        # self.presorted = presorted
        with g.local_scope():
            g.srcdata["h"] = feat
            if norm is not None:
                g.edata["norm"] = norm
            # g.edata['etype'] = etypes
            # message passing
            from functools import partial

            update_dict = {
                etype: (partial(message, g=g), fn.sum("m", "h")) for etype in g.canonical_etypes
            }
            g.multi_update_all(etype_dict=update_dict, cross_reducer="sum")
            # g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[: g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
