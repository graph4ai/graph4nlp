import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.pytorch.linear import TypedLinear
import dgl.nn as dglnn
import typing as tp

from .base import GNNBase, GNNLayerBase
from ...data import GraphData, from_dgl

# The implementation of RGCN is copied from DGL
class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases,
        *,
        weight=True,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {
                f"rel_{rel}": dglnn.GraphConv(in_feat, out_feat, norm="right", weight=False, bias=False)
                for rel in range(num_rels)
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < self.num_rels and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, self.num_rels)
            else:
                self.weight = nn.Parameter(torch.Tensor(self.num_rels, in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))

        self.dropout = nn.Dropout(dropout)
        self.etype_map = {}

    def forward(self, g: dgl.DGLHeteroGraph, inputs: tp.Dict[str, torch.Tensor]):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        
        # def create_new_graph():
            
        
        new_canonical_etypes = []
        new_etypes = []
        for src_type, edge_type, dst_type in g.canonical_etypes:
            new_edge_type = self.etype_map.setdefault(edge_type, f"rel_{len(self.etype_map)}")
            new_canonical_etypes.append((src_type, new_edge_type, dst_type))
            new_etypes.append(new_edge_type)
        g._etypes = new_etypes
        g._canonical_etypes = new_canonical_etypes
        g._etype2canonical = {etype: canonical_etype for etype, canonical_etype in zip(new_etypes, new_canonical_etypes)}
        g._etypes_invmap = {canonical_etype: i for i, canonical_etype in enumerate(new_canonical_etypes)}
        
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                f"rel_{i}": {"weight": w.squeeze(0)}
                for i, w in enumerate(torch.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


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
    rel_names : List[str]
        List of relation names.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    feat_drop : float, optional
        dropout rate. Default: ``0.0``
    """

    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        num_rels=None,
        direction_option=None,
        bias=True,
        activation=None,
        self_loop=True,
        feat_drop=0.0,
        regularizer="none",
        num_bases=4,
    ):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.num_rels = num_rels
        self.self_loop = self_loop
        self.feat_drop = feat_drop
        self.direction_option = direction_option
        self.activation = activation
        self.bias = bias
        self.RGCN_layers = nn.ModuleList()
        self.regularizer = regularizer
        self.num_basis = num_bases
        
        # if isinstance(self.num_rels, int):
        #     self.num_rels = [str(i) for i in range(self.num_rels)]

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
                    direction_option=self.direction_option,
                    bias=self.bias,
                    activation=self.activation,
                    self_loop=self.self_loop,
                    feat_drop=self.feat_drop,
                    regularizer=regularizer,
                    num_bases=num_bases,
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
                    direction_option=self.direction_option,
                    bias=self.bias,
                    activation=self.activation,
                    self_loop=self.self_loop,
                    feat_drop=self.feat_drop,
                    regularizer=regularizer,
                    num_bases=num_bases,
                )
            )
        # output projection
        self.RGCN_layers.append(
            RGCNLayer(
                hidden_size[-1] if self.num_layers > 1 else input_size,
                output_size,
                num_rels=self.num_rels,
                direction_option=self.direction_option,
                bias=self.bias,
                activation=self.activation,
                self_loop=self.self_loop,
                feat_drop=self.feat_drop,
                regularizer=regularizer,
                num_bases=num_bases,
            )
        )
        # Print named parameters
        # for k, v in self.named_parameters():
        #     print(f'{k}: {v}')

    def forward(self, graph: GraphData):
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
        # feat = graph.node_features["node_feat"]
        # if self.direction_option == "bi_sep":
        #     h = [feat, feat]
        # else:
        #     h = feat

        # get the node feature tensor from graph
        g = graph.to_dgl()  # transfer the current NLPgraph to DGL graph
        h: torch.Tensor = g.ndata["node_feat"]
        
        # Make node feature dictionary
        feat_dict: tp.Dict[str, torch.Tensor] = {}
        import numpy as np
        node_types = np.array(graph.ntypes,)
        for i in range(max(node_types) + 1):
            index = torch.tensor(np.where(node_types == i)[0], device=graph.device)
            feat_dict[i] = torch.index_select(h, 0, index)
        
        # output projection
        if self.num_layers > 1:
            for l in range(0, self.num_layers - 1):
                h = self.RGCN_layers[l](g, feat_dict)

        h = self.RGCN_layers[-1](g, h)

        if self.direction_option == "bi_sep":
            logits = torch.cat(logits, -1)

        # Unpack node feature dictionary
        if len(g.ntypes) == 1:
            h = h[0]
        g.ndata["node_emb"] = h  # put the results into the NLPGraph
        graph_data = from_dgl(g=g)
        if graph.batch is not None:
            graph_data.copy_batch_info(graph)
        return graph_data


class RGCNLayer(GNNLayerBase):
    r"""A wrapper for RGCNLayer.

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
    feat_drop : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_rels,
        direction_option=None,
        bias=True,
        activation=None,
        self_loop=False,
        feat_drop=0.0,
        layer_norm=False,
        regularizer=None,
        num_bases=None,
    ):
        super(RGCNLayer, self).__init__()
        if direction_option == "undirected":
            self.model = UndirectedRGCNLayer(
                input_size,
                output_size,
                num_rels=num_rels,
                bias=bias,
                activation=activation,
                self_loop=self_loop,
                feat_drop=feat_drop,
                layer_norm=layer_norm,
                regularizer=regularizer,
                num_bases=num_bases,
            )
        elif direction_option == "bi_sep":
            self.model = BiSepRGCNLayer(
                input_size,
                output_size,
                num_rels=num_rels,
                bias=bias,
                activation=activation,
                self_loop=self_loop,
                feat_drop=feat_drop,
                layer_norm=layer_norm,
                regularizer=regularizer,
                num_bases=num_bases,
            )
        elif direction_option == "bi_fuse":
            self.model = BiFuseRGCNLayer(
                input_size,
                output_size,
                num_rels=num_rels,
                bias=bias,
                activation=activation,
                self_loop=self_loop,
                feat_drop=feat_drop,
                layer_norm=layer_norm,
                regularizer=regularizer,
                num_bases=num_bases,
            )
        else:
            raise RuntimeError("Unknown `direction_option` value: {}".format(direction_option))

    def forward(self, graph: dgl.DGLHeteroGraph, feat: tp.Dict[str, torch.Tensor]):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        return self.model(graph, feat)


class UndirectedRGCNLayer(GNNLayerBase):
    r"""An undirected RGCN layer.

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
    feat_drop : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_rels,
        bias=True,
        activation=None,
        self_loop=False,
        feat_drop=0.0,
        layer_norm=False,
        regularizer=None,
        num_bases=None,
        dropout=0.0,
        **kwargs,
    ):
        super(UndirectedRGCNLayer, self).__init__()
        self.layer = RelGraphConvLayer(
            in_feat=input_size,
            out_feat=output_size,
            num_rels=num_rels,
            num_bases=num_bases,
            activation=activation,
            self_loop=self_loop,
            dropout=dropout,
        )

    def forward(self, g: dgl.DGLHeteroGraph, feat: tp.Dict[str, torch.Tensor]):
        return self.layer(g, feat)


class BiFuseRGCNLayer(GNNLayerBase):
    r"""A Bidirectional version for RGCNLayer, with an additional fuse layer.

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
    feat_drop : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``
    regularizer: str, optional
        Which weight regularizer to use "basis" or "bdd":
            - "basis" is short for basis-decomposition.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_rels,
        bias=True,
        activation=None,
        self_loop=False,
        feat_drop=0.0,
        layer_norm=False,
        regularizer=None,
        num_bases=None,
    ):
        super(BiFuseRGCNLayer, self).__init__()
        self.ln_fwd = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)
        self.ln_bwd = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)

        # self.linear_dict_forward = nn.ModuleDict(
        #     {str(i): nn.Linear(input_size, output_size, bias=bias) for i in range(num_rels)}
        # )
        # self.linear_dict_backward = nn.ModuleDict(
        #     {str(i): nn.Linear(input_size, output_size, bias=bias) for i in range(num_rels)}
        # )

        # self.linear_r = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias_forward = nn.Parameter(torch.Tensor(output_size))
            nn.init.zeros_(self.h_bias_forward)
            self.h_bias_backward = nn.Parameter(torch.Tensor(output_size))
            nn.init.zeros_(self.h_bias_backward)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight_forward = nn.LayerNorm(output_size, elementwise_affine=True)
            self.layer_norm_weight_backward = nn.LayerNorm(output_size, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight_forward = nn.Parameter(torch.Tensor(input_size, output_size))
            nn.init.xavier_uniform_(self.loop_weight_forward, gain=nn.init.calculate_gain("relu"))

            self.loop_weight_backward = nn.Parameter(torch.Tensor(input_size, output_size))
            nn.init.xavier_uniform_(self.loop_weight_backward, gain=nn.init.calculate_gain("relu"))

        self.fuse_linear = nn.Linear(4 * output_size, output_size, bias=True)
        self.dropout = nn.Dropout(feat_drop)

    def forward(self, g: dgl.DGLHeteroGraph, feat: torch.Tensor, norm=None):
        def message(edges, g, direction):
            """Message function."""
            # linear_dict = (
            #     self.linear_dict_forward if direction == "forward" else self.linear_dict_backward
            # )
            # ln = linear_dict[str(g.canonical_etypes.index(edges._etype))]
            # m = ln(edges.src["h"])

            ln = self.ln_fwd if direction == "forward" else self.ln_bwd
            etypes = torch.tensor(
                [g.canonical_etypes.index(edges._etype)] * edges.src["h"].shape[0]
            ).to(edges.src["h"].device)
            m = ln(edges.src["h"], etypes)
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
                etype: (partial(message, g=g, direction="forward"), fn.sum("m", "h"))
                for etype in g.canonical_etypes
            }
            g.multi_update_all(etype_dict=update_dict, cross_reducer="sum")
            # g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_weight_forward(h)
            if self.bias:
                h = h + self.h_bias_forward
            if self.self_loop:
                h = h + feat[: g.num_dst_nodes()] @ self.loop_weight_forward
            h_forward = h

        g = g.reverse()
        with g.local_scope():
            g.srcdata["h"] = feat
            if norm is not None:
                g.edata["norm"] = norm
            # g.edata['etype'] = etypes
            # message passing
            from functools import partial

            update_dict = {
                etype: (partial(message, g=g, direction="backward"), fn.sum("m", "h"))
                for etype in g.canonical_etypes
            }
            g.multi_update_all(etype_dict=update_dict, cross_reducer="sum")
            # g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_weight_backward(h)
            if self.bias:
                h = h + self.h_bias_backward
            if self.self_loop:
                h = h + feat[: g.num_dst_nodes()] @ self.loop_weight_backward
            h_backward = h

        fuse_vector = torch.cat(
            [h_forward, h_backward, h_forward * h_backward, h_forward - h_backward], dim=-1
        )
        fuse_gate_vector = torch.sigmoid(self.fuse_linear(fuse_vector))
        h = fuse_gate_vector * h_forward + (1 - fuse_gate_vector) * h_backward

        if self.activation:
            h = self.activation(h)
        h = self.dropout(h)
        return h


class BiSepRGCNLayer(GNNLayerBase):
    r"""A Bidirectional version for RGCNLayer.

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
    feat_drop : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``
    regularizer: str, optional
        Which weight regularizer to use "basis" or "bdd":
            - "basis" is short for basis-decomposition.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_rels,
        bias=True,
        activation=None,
        self_loop=False,
        feat_drop=0.0,
        layer_norm=False,
        regularizer=None,
        num_bases=None,
    ):
        super(BiSepRGCNLayer, self).__init__()
        self.ln_fwd = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)
        self.ln_bwd = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)

        # self.linear_dict_forward = nn.ModuleDict(
        #     {str(i): nn.Linear(input_size, output_size, bias=bias) for i in range(num_rels)}
        # )
        # self.linear_dict_backward = nn.ModuleDict(
        #     {str(i): nn.Linear(input_size, output_size, bias=bias) for i in range(num_rels)}
        # )

        # self.linear_r = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias_forward = nn.Parameter(torch.Tensor(output_size))
            nn.init.zeros_(self.h_bias_forward)
            self.h_bias_backward = nn.Parameter(torch.Tensor(output_size))
            nn.init.zeros_(self.h_bias_backward)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight_forward = nn.LayerNorm(output_size, elementwise_affine=True)
            self.layer_norm_weight_backward = nn.LayerNorm(output_size, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight_forward = nn.Parameter(torch.Tensor(input_size, output_size))
            nn.init.xavier_uniform_(self.loop_weight_forward, gain=nn.init.calculate_gain("relu"))

            self.loop_weight_backward = nn.Parameter(torch.Tensor(input_size, output_size))
            nn.init.xavier_uniform_(self.loop_weight_backward, gain=nn.init.calculate_gain("relu"))

        self.dropout = nn.Dropout(feat_drop)

    def forward(self, g: dgl.DGLHeteroGraph, feat: torch.Tensor, norm=None):
        def message(edges, g, direction):
            """Message function."""
            # linear_dict = (
            #     self.linear_dict_forward if direction == "forward" else self.linear_dict_backward
            # )
            # ln = linear_dict[str(g.canonical_etypes.index(edges._etype))]
            ln = self.ln_fwd if direction == "forward" else self.ln_bwd
            etypes = torch.tensor(
                [g.canonical_etypes.index(edges._etype)] * edges.src["h"].shape[0]
            ).to(edges.src["h"].device)
            m = ln(edges.src["h"], etypes)
            if "norm" in edges.data:
                m = m * edges.data["norm"]
            return {"m": m}

        feat_forward, feat_backward = feat
        # self.presorted = presorted
        with g.local_scope():
            g.srcdata["h"] = feat_forward
            if norm is not None:
                g.edata["norm"] = norm
            # g.edata['etype'] = etypes
            # message passing
            from functools import partial

            update_dict = {
                etype: (partial(message, g=g, direction="forward"), fn.sum("m", "h"))
                for etype in g.canonical_etypes
            }
            g.multi_update_all(etype_dict=update_dict, cross_reducer="sum")
            # g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_weight_forward(h)
            if self.bias:
                h = h + self.h_bias_forward
            if self.self_loop:
                h = h + feat_forward[: g.num_dst_nodes()] @ self.loop_weight_forward
            h_forward = h

        g = g.reverse()
        with g.local_scope():
            g.srcdata["h"] = feat_backward
            if norm is not None:
                g.edata["norm"] = norm
            # g.edata['etype'] = etypes
            # message passing
            from functools import partial

            update_dict = {
                etype: (partial(message, g=g, direction="backward"), fn.sum("m", "h"))
                for etype in g.canonical_etypes
            }
            g.multi_update_all(etype_dict=update_dict, cross_reducer="sum")
            # g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_weight_backward(h)
            if self.bias:
                h = h + self.h_bias_backward
            if self.self_loop:
                h = h + feat_backward[: g.num_dst_nodes()] @ self.loop_weight_backward
            h_backward = h

        if self.activation:
            h_forward = self.activation(h_forward)
            h_backward = self.activation(h_backward)
        h_forward = self.dropout(h_forward)
        h_backward = self.dropout(h_backward)
        return [h_forward, h_backward]
