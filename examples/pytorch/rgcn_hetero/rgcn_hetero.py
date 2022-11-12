from re import S
import warnings
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.nn.pytorch import HeteroGraphConv, GraphConv

from graph4nlp.pytorch.modules.graph_embedding_learning.base import GNNBase, GNNLayerBase


class RGCNHetero(GNNBase):
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
    rels_names : List[str]
        Names of relation(edge) types.
    node_types: List[str]
        Names of node types.
    num_nodes: int,
        Number of nodes.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``-1``.
    use_self_loop : bool, optional
        True to include self loop message. Default: ``False``.
    gpu: int, optional
        GPU device number. Default: ``-1`` (CPU)
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    """

    def __init__(
        self,
        num_hidden_layers,
        input_size,
        hidden_size,
        output_size,
        rel_names,
        node_types,
        num_nodes,
        num_bases=-1,
        use_self_loop=False,
        gpu=-1,
        dropout=0.0
    ):
        super(RGCNHetero, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.num_nodes = num_nodes
        self.use_self_loop = use_self_loop
        self.dropout = dropout
        self.gpu = gpu

        self.embs = nn.ParameterDict({})
        for nt in node_types:
            embed = nn.Parameter(torch.Tensor(num_nodes[nt], hidden_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embs[nt] = embed

        self.RGCN_layers = nn.ModuleList()

        # hidden layers
        for l in range(self.num_hidden_layers):
            # due to multi-head, the input_size = hidden_size * num_heads
            self.RGCN_layers.append(
                RGCNLayerHetero(
                    hidden_size,
                    hidden_size,
                    rel_names=rel_names,
                    num_bases=self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout
                )
            )
        # output projection
        self.RGCN_layers.append(
            RGCNLayerHetero(
                hidden_size,
                output_size,
                rel_names=self.rel_names,
                num_bases=self.num_bases,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=self.dropout
            )
        )
        self.h = self.embs
        if self.gpu != -1:
            self.to(device=self.gpu)

    def forward(self, graph, h=None):
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
        if h is None:
            h = self.embs
        for l in range(self.num_hidden_layers):
            h = self.RGCN_layers[l](g, h)
        logits = self.RGCN_layers[-1](g, h)
        
        # graph.node_features['node_feat'] = h {'type1': (num_node_type1 x emb_dim), 'type2': (num_node_type2 x emb_dim)}
        # graph.node_features["node_emb"] = logits  # put the results into the NLPGraph
        return logits


class RGCNLayerHetero(GNNLayerBase):
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
        rel_names,
        num_bases=None,
        activation=None,
        self_loop=False,
        dropout=0.0
    ):
        super(RGCNLayerHetero, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rel_names = rel_names
        self.num_bases = num_bases
        # self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.dropout = dropout

        self.conv = HeteroGraphConv(
            {
                rel: GraphConv(input_size, output_size, norm='right', weight=False, bias=False) for rel in rel_names
            }
        )
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, graph, inputs):
        """

        Parameters:
        ----------
        graph: DGLHeteroGraph
            The graph
        inputs: dict[str, torch.Tensor]
            New node features for each node type
        """
        graph = graph.local_var()
        inputs_src = inputs_dst = inputs

        hs = self.conv(graph, inputs)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
