import dgl.function as fn
import torch
import torch.nn as nn
from dgl.base import DGLError
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.utils import expand_as_pair

from ..utils.generic_utils import Identity
from .base import GNNBase, GNNLayerBase


class GAT(GNNBase):
    r"""Multi-layer Graph Attention Network (GAT).
    Support both `unidirectional GAT
    <https://arxiv.org/abs/1710.10903>`__ and bidirectional versions
    including `GAT-BiSep <https://arxiv.org/abs/1808.07624>`__ and
    `GAT-BiFuse <https://arxiv.org/abs/1908.04942>`__.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    num_layers: int
        Number of GAT layers.
    input_size : int, or pair of ints
        Input feature size.
        If the layer is to be applied to a unidirectional bipartite graph, ``input_size``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    hidden_size: int or list of int
        Hidden size per GAT layer. If ``int`` is given, all layers are forced to have
        the same hidden size.
    output_size : int
        Output feature size.
    heads : int or list of int
        Number of heads per GAT layer. If ``int`` is given, all layers are forced to
        have the same number of heads.
    direction_option: str
        Whether to use unidirectional (i.e., regular "undirected") or bidirectional (i.e., "bi_sep"
        and "bi_fuse") versions.
        Default : ``'bi_sep'``.
    feat_drop : float, optional
        Dropout rate on feature, default: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, default: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope, default: ``0.2``.
    residual : bool, optional
        If True, use residual connection.
        Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    """

    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        heads,
        direction_option="bi_sep",
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.direction_option = direction_option
        self.gat_layers = nn.ModuleList()
        assert self.num_layers > 0
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * (self.num_layers - 1)

        if isinstance(heads, int):
            heads = [heads] * self.num_layers

        if self.num_layers > 1:
            # input projection
            self.gat_layers.append(
                GATLayer(
                    input_size,
                    hidden_size[0],
                    heads[0],
                    direction_option=self.direction_option,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=activation,
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )

        # hidden layers
        for l in range(1, self.num_layers - 1):
            # due to multi-head, the input_size = hidden_size * num_heads
            self.gat_layers.append(
                GATLayer(
                    hidden_size[l - 1] * heads[l - 1],
                    hidden_size[l],
                    heads[l],
                    direction_option=self.direction_option,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=activation,
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
        # output projection
        self.gat_layers.append(
            GATLayer(
                hidden_size[-1] * heads[-2] if self.num_layers > 1 else input_size,
                output_size,
                heads[-1],
                direction_option=self.direction_option,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=None,
                allow_zero_in_degree=allow_zero_in_degree,
            )
        )

    def forward(self, graph):
        r"""Compute multi-layer graph attention network.

        Parameters
        ----------
        graph : GraphData
            The graph data containing topology and features.

        Returns
        -------
        GraphData
            The output graph data containing updated embeddings.
        """
        feat = graph.node_features["node_feat"]
        dgl_graph = graph.to_dgl()

        if self.direction_option == "bi_sep":
            h = [feat, feat]
        else:
            h = feat

        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](dgl_graph, h)
            if self.direction_option == "bi_sep":
                h = [each.flatten(1) for each in h]
            else:
                h = h.flatten(1)

        # output projection
        logits = self.gat_layers[-1](dgl_graph, h)

        if self.direction_option == "bi_sep":
            logits = [each.mean(1) for each in logits]
            logits = torch.cat(logits, -1)
        else:
            logits = logits.mean(1)

        graph.node_features["node_emb"] = logits

        return graph


class GATLayer(GNNLayerBase):
    r"""Single-layer Graph Attention Network (GAT).
    Support both `unidirectional GAT
    <https://arxiv.org/abs/1710.10903>`__ and bidirectional versions
    including `GAT-BiSep <https://arxiv.org/abs/1808.07624>`__ and
    `GAT-BiFuse <https://arxiv.org/abs/1908.04942>`__.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

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
    num_heads : int
        Number of heads in Multi-Head Attention.
    direction_option: str
        Whether use unidirectional (i.e., regular "undirected") or bidirectional (i.e., `bi_sep`
        and `bi_fuse`) versions.
        Default: ``'bi_sep'``.
    feat_drop : float, optional
        Dropout rate on feature, default: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, default: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope, default: ``0.2``.
    residual : bool, optional
        If True, use residual connection.
        Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree: bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_heads,
        direction_option="bi_sep",
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GATLayer, self).__init__()
        if num_heads > 1 and residual:
            residual = False
            import warnings

            warnings.warn("The residual option must be False when num_heads > 1")
        if direction_option == "undirected":
            self.model = UndirectedGATLayerConv(
                input_size,
                output_size,
                num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
            )
        elif direction_option == "bi_sep":
            self.model = BiSepGATLayerConv(
                input_size,
                output_size,
                num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
            )
        elif direction_option == "bi_fuse":
            self.model = BiFuseGATLayerConv(
                input_size,
                output_size,
                num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
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


class UndirectedGATLayerConv(GNNLayerBase):
    r"""Apply `Graph Attention Network <https://arxiv.org/abs/1710.10903>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

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
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, default: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, default: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope, default: ``0.2``.
    residual : bool, optional
        If True, use residual connection.
        Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree: bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(UndirectedGATLayerConv, self).__init__()
        self.model = GATConv(
            in_feats=input_size,
            out_feats=output_size,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
        )

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


class BiFuseGATLayerConv(GNNLayerBase):
    # TODO: improve math descriptions bidirectional GNN.
    r"""Apply `Bidirectional Fuse GNN mechanism <https://arxiv.org/abs/1908.04942>`__
    to `Graph Attention Network <https://arxiv.org/abs/1710.10903>`__ over an
    input signal. Fuse aggregated embeddings from both incoming and outgoing
    directions before updating node embeddings.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

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
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, default: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, default: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope, default: ``0.2``.
    residual : bool, optional
        If True, use residual connection.
        Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(BiFuseGATLayerConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(input_size)
        self._out_feats = output_size
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(input_size, tuple):
            self.fc_src_fw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)
            self.fc_dst_fw = nn.Linear(self._in_dst_feats, output_size * num_heads, bias=False)

            self.fc_src_bw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)
            self.fc_dst_bw = nn.Linear(self._in_dst_feats, output_size * num_heads, bias=False)
        else:
            self.fc_fw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)
            self.fc_bw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)

        self.attn_l_fw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.attn_l_bw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.attn_r_fw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.attn_r_bw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu_fw = nn.LeakyReLU(negative_slope)
        self.leaky_relu_bw = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != output_size:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * output_size, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation

        self.fuse_linear = nn.Linear(4 * output_size, output_size, bias=True)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc_fw") or hasattr(self, "fc_bw"):
            nn.init.xavier_normal_(self.fc_fw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_bw.weight, gain=gain)
        else:  # bipartite graph neural works
            nn.init.xavier_normal_(self.fc_src_fw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_src_bw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst_fw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst_bw.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l_fw, gain=gain)
        nn.init.xavier_normal_(self.attn_l_bw, gain=gain)
        nn.init.xavier_normal_(self.attn_r_fw, gain=gain)
        nn.init.xavier_normal_(self.attn_r_bw, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

        if hasattr(self, "fuse_linear"):
            nn.init.xavier_normal_(self.fuse_linear.weight, gain=gain)

    def forward(self, graph, feat):
        """Parameters
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
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

        feat_fw = feat_bw = feat

        if isinstance(feat_fw, tuple):
            h_src_fw = self.feat_drop(feat_fw[0])
            h_dst_fw = self.feat_drop(feat_fw[1])
        else:
            h_src_fw = h_dst_fw = self.feat_drop(feat_fw)

        if isinstance(feat_bw, tuple):
            h_src_bw = self.feat_drop(feat_bw[0])
            h_dst_bw = self.feat_drop(feat_bw[1])
        else:
            h_src_bw = h_dst_bw = self.feat_drop(feat_bw)

        # forward direction
        with graph.local_scope():
            if isinstance(feat_fw, tuple):
                feat_src = self.fc_src_fw(h_src_fw).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst_fw(h_dst_fw).view(-1, self._num_heads, self._out_feats)
            else:
                feat_src = feat_dst = self.fc_fw(h_src_fw).view(
                    -1, self._num_heads, self._out_feats
                )
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l_fw).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r_fw).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu_fw(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            agg_emb_fw = graph.dstdata["ft"]

        # backward direction
        graph = graph.reverse()
        with graph.local_scope():
            if isinstance(feat_bw, tuple):
                feat_src = self.fc_src_bw(h_src_bw).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst_bw(h_dst_bw).view(-1, self._num_heads, self._out_feats)
            else:
                feat_src = feat_dst = self.fc_bw(h_src_bw).view(
                    -1, self._num_heads, self._out_feats
                )

            el = (feat_src * self.attn_l_bw).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r_bw).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu_bw(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            agg_emb_bw = graph.dstdata["ft"]

        fuse_vector = torch.cat(
            [agg_emb_fw, agg_emb_bw, agg_emb_fw * agg_emb_bw, agg_emb_fw - agg_emb_bw], dim=-1
        )
        fuse_gate_vector = torch.sigmoid(self.fuse_linear(fuse_vector))
        agg_emb = fuse_gate_vector * agg_emb_fw + (1 - fuse_gate_vector) * agg_emb_bw

        # residual
        if self.res_fc is not None:
            h_dst = h_dst_fw
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            agg_emb = agg_emb + resval

        # activation
        if self.activation:
            agg_emb = self.activation(agg_emb)

        return agg_emb


class BiSepGATLayerConv(GNNLayerBase):
    # TODO: improve math descriptions bidirectional GNN.
    r"""Apply `Bidirectional Separate GNN mechanism <https://arxiv.org/abs/1808.07624>`__
    to `Graph Attention Network <https://arxiv.org/abs/1710.10903>`__ over an
    input signal. Compute node embeddings for incoming and outgoing directions
    separately, and then concatenate the two output node embeddings
    after the final layer.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

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
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, default: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, default: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope, default: ``0.2``.
    residual : bool, optional
        If True, use residual connection.
        Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(BiSepGATLayerConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(input_size)
        self._out_feats = output_size
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(input_size, tuple):
            self.fc_src_fw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)
            self.fc_dst_fw = nn.Linear(self._in_dst_feats, output_size * num_heads, bias=False)

            self.fc_src_bw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)
            self.fc_dst_bw = nn.Linear(self._in_dst_feats, output_size * num_heads, bias=False)
        else:
            self.fc_fw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)
            self.fc_bw = nn.Linear(self._in_src_feats, output_size * num_heads, bias=False)

        self.attn_l_fw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.attn_l_bw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.attn_r_fw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.attn_r_bw = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_size)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu_fw = nn.LeakyReLU(negative_slope)
        self.leaky_relu_bw = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != output_size:
                self.res_fc_fw = nn.Linear(self._in_dst_feats, num_heads * output_size, bias=False)
                self.res_fc_bw = nn.Linear(self._in_dst_feats, num_heads * output_size, bias=False)
            else:
                self.res_fc_fw = self.res_fc_bw = Identity()
        else:
            self.register_buffer("res_fc_fw", None)
            self.register_buffer("res_fc_bw", None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc_fw") or hasattr(self, "fc_bw"):
            nn.init.xavier_normal_(self.fc_fw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_bw.weight, gain=gain)
        else:  # bipartite graph neural works
            nn.init.xavier_normal_(self.fc_src_fw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_src_bw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst_fw.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst_bw.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l_fw, gain=gain)
        nn.init.xavier_normal_(self.attn_l_bw, gain=gain)
        nn.init.xavier_normal_(self.attn_r_fw, gain=gain)
        nn.init.xavier_normal_(self.attn_r_bw, gain=gain)

        if isinstance(self.res_fc_fw, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_fw.weight, gain=gain)

        if isinstance(self.res_fc_bw, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_bw.weight, gain=gain)

        if hasattr(self, "fuse_linear"):
            nn.init.xavier_normal_(self.fuse_linear.weight, gain=gain)

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat:
            A list/tuple containing incoming feature ``feat_fw``
            and outgoing feature ``feat_bw``.
            - ``feat_fw`` : torch.Tensor or pair of torch.Tensor
                    If a torch.Tensor is given, the input feature of
                    shape :math:`(N, D_{in})` where :math:`D_{in}` is
                    size of input feature, :math:`N` is the number of
                    nodes. If a pair of torch.Tensor is given, the pair
                    must contain two tensors of shape :math:`(N_{in},
                    D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
            - ``feat_bw``: torch.Tensor or pair of torch.Tensor
                    If a torch.Tensor is given, the input feature of
                    shape :math:`(N, D_{in})` where :math:`D_{in}` is
                    size of input feature, :math:`N` is the number of
                    nodes. If a pair of torch.Tensor is given, the pair
                    must contain two tensors of shape :math:`(N_{in},
                    D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        Pair of torch.Tensor
            Each output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

        feat_fw, feat_bw = feat

        if isinstance(feat_fw, tuple):
            h_src_fw = self.feat_drop(feat_fw[0])
            h_dst_fw = self.feat_drop(feat_fw[1])
        else:
            h_src_fw = h_dst_fw = self.feat_drop(feat_fw)

        if isinstance(feat_bw, tuple):
            h_src_bw = self.feat_drop(feat_bw[0])
            h_dst_bw = self.feat_drop(feat_bw[1])
        else:
            h_src_bw = h_dst_bw = self.feat_drop(feat_bw)

        # forward direction
        with graph.local_scope():
            if isinstance(feat_fw, tuple):
                feat_src = self.fc_src_fw(h_src_fw).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst_fw(h_dst_fw).view(-1, self._num_heads, self._out_feats)
            else:
                feat_src = feat_dst = self.fc_fw(h_src_fw).view(
                    -1, self._num_heads, self._out_feats
                )
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l_fw).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r_fw).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu_fw(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            agg_emb_fw = graph.dstdata["ft"]

        # backward direction
        graph = graph.reverse()
        with graph.local_scope():
            if isinstance(feat_bw, tuple):
                feat_src = self.fc_src_bw(h_src_bw).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst_bw(h_dst_bw).view(-1, self._num_heads, self._out_feats)
            else:
                feat_src = feat_dst = self.fc_bw(h_src_bw).view(
                    -1, self._num_heads, self._out_feats
                )

            el = (feat_src * self.attn_l_bw).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r_bw).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu_bw(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            agg_emb_bw = graph.dstdata["ft"]

        # residual
        if self.res_fc_fw is not None:
            resval_fw = self.res_fc_fw(h_dst_fw).view(h_dst_fw.shape[0], -1, self._out_feats)
            resval_bw = self.res_fc_bw(h_dst_bw).view(h_dst_bw.shape[0], -1, self._out_feats)
            agg_emb_fw = agg_emb_fw + resval_fw
            agg_emb_bw = agg_emb_bw + resval_bw

        # activation
        if self.activation:
            agg_emb_fw = self.activation(agg_emb_fw)
            agg_emb_bw = self.activation(agg_emb_bw)

        return [agg_emb_fw, agg_emb_bw]
