import warnings
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.nn.pytorch import RelGraphConv

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
    hidden_size: int
        Hidden layer size.
    output_size : int
        Output feature size.
    num_rels : int
        Number of relations.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``-1`` [all].
    use_self_loop : bool, optional
        True to include self loop message. Default: ``False``.
    gpu : int, optional
        True to use gpu. Default: ``-1`` [cpu].
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
        num_bases=-1,
        use_self_loop=True,
        gpu=False,
        dropout=0.0,

    ):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        if num_bases == -1:
            num_bases = num_rels
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.use_self_loop = use_self_loop
        self.dropout = nn.Dropout(dropout)
        self.gpu = gpu

        self.RGCN_layers = nn.ModuleList()
        # input layers:
        self.RGCN_layers.append(
            RGCNLayer(
                input_size,
                hidden_size,
                num_rels=self.num_rels,
                regularizer="basis",
                num_bases=self.num_bases,
                bias=True,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=dropout
            )
        )
        if self.num_layers > 1:
            # input projection
            self.RGCN_layers.append(
                RGCNLayer(
                    hidden_size,
                    hidden_size,
                    num_rels=self.num_rels,
                    regularizer="basis",
                    num_bases=self.num_bases,
                    bias=True,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=dropout
                )
            )
        
        # hidden layers
        for l in range(1, self.num_layers-1):
            self.RGCN_layers.append(
                RGCNLayer(
                    hidden_size,
                    hidden_size,
                    num_rels=self.num_rels,
                    regularizer="basis",
                    num_bases=self.num_bases,
                    bias=True,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=dropout
                )
            )
        # output projection
        self.RGCN_layers.append(
            RGCNLayer(
                hidden_size,
                output_size,
                num_rels=self.num_rels,
                regularizer="basis",
                num_bases=self.num_bases,
                bias=True,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=dropout
            )
        )

        if self.gpu != -1:
            self.to(device=self.gpu)

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

        # transfer the current NLPgraph to DGL graph
        g = graph.to_dgl()
        h = graph.node_features['node_feat']
        edge_type = g.edata['edge__TYPE'].long()
        for l in range(self.num_layers):
            h = self.RGCN_layers[l](g, h, edge_type, g.edata['edge_norm'])
            h = self.dropout(F.relu(h))
        logits = self.RGCN_layers[-1](g, h, edge_type, g.edata['edge_norm'])
        
        # put the results into the NLPGraph
        # graph.node_features['node_feat'] = h
        graph.node_features["node_emb"] = logits  

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
        num_bases=-1,
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
