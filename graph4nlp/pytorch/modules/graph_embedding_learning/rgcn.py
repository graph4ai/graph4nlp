import dgl
import dgl.function as fn
from dgl.nn.pytorch.linear import TypedLinear
import torch
import torch.nn as nn

from .base import GNNBase, GNNLayerBase


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
        num_rels,
        direction_option=None,
        bias=True,
        activation=None,
        self_loop=True,
        feat_drop=0.0,
        regularizer='basis',
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
        feat = graph.node_features["node_feat"]
        if self.direction_option == "bi_sep":
            h = [feat, feat]
        else:
            h = feat

        # get the node feature tensor from graph
        g = graph.to_dgl()  # transfer the current NLPgraph to DGL graph
        # edge_type = g.edata[dgl.ETYPE].long()
        # output projection
        if self.num_layers > 1:
            for l in range(0, self.num_layers - 1):
                h = self.RGCN_layers[l](g, h)

        logits = self.RGCN_layers[-1](g, h)

        if self.direction_option == "bi_sep":
            logits = torch.cat(logits, -1)

        graph.node_features["node_emb"] = logits  # put the results into the NLPGraph
        return graph


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
        num_bases=None
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

    def forward(self, graph, feat):
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
    ):
        super(UndirectedRGCNLayer, self).__init__()
        # self.linear_dict = nn.ModuleDict({
        #     str(i): nn.Linear(input_size, output_size, bias=bias) for i in range(num_rels)
        # })
        self.linear = TypedLinear(
            in_size=input_size,
            out_size=output_size,
            num_types=num_rels,
            regularizer=regularizer,
            num_bases=num_bases,
        )
        # self.linear_r = TypedLinear(input_size, output_size, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(output_size))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(output_size, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(input_size, output_size))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))

        self.dropout = nn.Dropout(feat_drop)

    def forward(self, g: dgl.DGLHeteroGraph, feat: torch.Tensor, norm=None):
        def message(edges, g):
            """Message function."""
            # ln = self.linear(edges.src['h'], edges.data['type'])
            # ln = self.linear_dict[str(g.canonical_etypes.index(edges._etype))]
            # m = ln(edges.src["h"])

            etypes = torch.tensor(
                [g.canonical_etypes.index(edges._etype)] * edges.src["h"].shape[0]
            ).to(edges.src["h"].device)
            m = self.linear(edges.src["h"], etypes)

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
        num_bases=None
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
