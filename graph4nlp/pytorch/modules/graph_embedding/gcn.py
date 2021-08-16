import dgl.function as fn
import torch
import torch.nn as nn
from torch.nn import init

from .base import GNNBase, GNNLayerBase


class GCN(GNNBase):
    r"""Multi-layer Graph Convolutional Networks (GCN).
    Support both `Unidirectional GCN
    <https://arxiv.org/pdf/1609.02907>`__ and bidirectional versions
    including `GCN-BiSep` and `GCN-BiFuse`.

    For the Unidirectional GCN,

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ij}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ij} = \sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`),
    and :math:`\sigma` is an activation function.

    Parameters
    ----------
    num_layers: int
        Number of GCN layers.

    input_size: int
        Input feature size of the first GCN layer.

    hidden_size: int
        Hidden size per GCN layer.

    output_size: int
        Output feature size of the final GCN layer.

    direction_option: str
        Whether to use unidirectional (i.e., regular) or bidirectional
        (i.e., "bi_sep" and "bi_fuse") versions.
        Default : ``'bi_sep'``.

    gcn_norm: str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.

    weight: bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.

    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    allow_zero_in_degree: bool, optional

    use_edge_weight: bool, optional
    """

    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        direction_option="bi_sep",
        feat_drop=0.0,
        gcn_norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        use_edge_weight=False,
        residual=True,
    ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.direction_option = direction_option
        self.gcn_layers = nn.ModuleList()
        assert self.num_layers > 0
        self.use_edge_weight = use_edge_weight

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * (self.num_layers - 1)

        if self.num_layers > 1:
            # input projection
            self.gcn_layers.append(
                GCNLayer(
                    input_size,
                    hidden_size[0],
                    direction_option=self.direction_option,
                    feat_drop=feat_drop,
                    gcn_norm=gcn_norm,
                    weight=weight,
                    bias=bias,
                    activation=activation,
                    allow_zero_in_degree=allow_zero_in_degree,
                    residual=residual,
                )
            )

        # hidden layers
        for l in range(1, self.num_layers - 1):
            # due to multi-head, the input_size = hidden_size * num_heads
            self.gcn_layers.append(
                GCNLayer(
                    hidden_size[l - 1],
                    hidden_size[l],
                    direction_option=self.direction_option,
                    feat_drop=feat_drop,
                    gcn_norm=gcn_norm,
                    weight=weight,
                    bias=bias,
                    activation=activation,
                    allow_zero_in_degree=allow_zero_in_degree,
                    residual=residual,
                )
            )
        # output projection
        self.gcn_layers.append(
            GCNLayer(
                hidden_size[-1] if self.num_layers > 1 else input_size,
                output_size,
                direction_option=self.direction_option,
                feat_drop=feat_drop,
                gcn_norm=gcn_norm,
                weight=weight,
                bias=bias,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
                residual=residual,
            )
        )

    def forward(self, graph):
        r"""Compute multi-layer graph convolutional networks.

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

        if self.use_edge_weight:
            edge_weight = graph.edge_features["edge_weight"]
            if self.direction_option != "undirected":
                reverse_edge_weight = graph.edge_features["reverse_edge_weight"]
            else:
                reverse_edge_weight = None
        else:
            edge_weight = None
            reverse_edge_weight = None

        for l in range(self.num_layers - 1):
            h = self.gcn_layers[l](
                dgl_graph, h, edge_weight=edge_weight, reverse_edge_weight=reverse_edge_weight
            )
            if self.direction_option == "bi_sep":
                h = [each.flatten(1) for each in h]
            else:
                h = h.flatten(1)

        # output projection
        logits = self.gcn_layers[-1](dgl_graph, h)

        if self.direction_option == "bi_sep":
            # logits = [each.mean(1) for each in logits]
            logits = torch.cat(logits, -1)
        else:
            pass

        graph.node_features["node_emb"] = logits

        return graph


class GCNLayer(GNNLayerBase):
    r"""Single-layer GCN.

    Parameters
    ----------
    num_layers: int
        Number of GCN layers.

    input_size: int
        Input feature size of the first GCN layer.

    hidden_size: int
        Hidden size per GCN layer.

    output_size: int
        Output feature size of the final GCN layer.

    direction_option: str
        Whether to use unidirectional (i.e., regular) or bidirectional
        (i.e., "bi_sep" and "bi_fuse") versions.
        Default : ``'bi_sep'``.

    gcn_norm: str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.

    weight: bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.

    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    """

    def __init__(
        self,
        input_size,
        output_size,
        direction_option="bi_sep",
        feat_drop=0.0,
        gcn_norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        residual=True,
    ):
        super(GCNLayer, self).__init__()
        if direction_option == "undirected":
            self.model = UndirectedGCNLayerConv(
                input_size,
                output_size,
                feat_drop=feat_drop,
                gcn_norm=gcn_norm,
                weight=weight,
                bias=bias,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
                residual=residual,
            )
        elif direction_option == "bi_sep":
            self.model = BiSepGCNLayerConv(
                input_size,
                output_size,
                feat_drop=feat_drop,
                gcn_norm=gcn_norm,
                weight=weight,
                bias=bias,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
                residual=residual,
            )
        elif direction_option == "bi_fuse":
            self.model = BiFuseGCNLayerConv(
                input_size,
                output_size,
                feat_drop=feat_drop,
                gcn_norm=gcn_norm,
                weight=weight,
                bias=bias,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
                residual=residual,
            )
        else:
            raise RuntimeError("Unknown `direction_option` value: {}".format(direction_option))

    def forward(self, graph, feat, weight=None, edge_weight=None, reverse_edge_weight=None):
        r"""Compute graph convolutional network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.

        feat : torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.

        weight: torch.Tensor, optional
            Optional external weight tensor.

        edge_weight: torch.Tensor
            Optional edge weight. edge_weight shape: "math:`(\text{num_edge}, 1)`.

        reverse_edge_weight: torch.Tensor
            Optional reverse edge weight. reverse_edge_weight shape: "math:`(\text{num_edge}, 1)`.
        """
        return self.model(graph, feat, weight, edge_weight, reverse_edge_weight)


class UndirectedGCNLayerConv(GNNLayerBase):
    r"""

    Description
    -----------
    Graph convolution was introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and mathematically is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ij}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ij} = \sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`),
    and :math:`\sigma` is an activation function.

    Parameters
    ----------
    input_size : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    output_size : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    gcn_norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(
        self,
        input_size,
        output_size,
        feat_drop=0.0,
        gcn_norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        residual=True,
    ):
        super(UndirectedGCNLayerConv, self).__init__()
        if gcn_norm not in ("none", "both", "right"):
            raise RuntimeError(
                'Invalid gcn_norm value. Must be either "none", "both" or "right".'
                ' But got "{}".'.format(gcn_norm)
            )
        self._input_size = input_size
        self._output_size = output_size
        self._gcn_norm = gcn_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._feat_drop = nn.Dropout(feat_drop)

        if weight:
            self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        if residual:
            if self._input_size != output_size:
                self.res_fc = nn.Linear(self._input_size, output_size, bias=True)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        self._activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None, reverse_edge_weight=None):
        r"""Compute graph convolution.

        Parameters
        ----------
        graph: DGLGraph
            The graph.

        feat: torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.

        weight: torch.Tensor, optional
            Optional external weight tensor.

        edge_weight: torch.Tensor
            Optional edge weight. edge_weight shape: "math:`(\text{num_edge}, 1)`.

        reverse_edge_weight: torch.Tensor
            Optional reverse edge weight. reverse_edge_weight shape: "math:`(\text{num_edge}, 1)`.
            For undirected GCN layer, reverse_edge_weight must be `None`.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        assert reverse_edge_weight is None
        graph = graph.local_var()

        feat_origin = feat
        feat = self._feat_drop(feat)

        if self._gcn_norm == "both":
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            gcn_norm = torch.pow(degs, -0.5)
            shp = gcn_norm.shape + (1,) * (feat.dim() - 1)
            gcn_norm = torch.reshape(gcn_norm, shp)
            feat = feat * gcn_norm

        if weight is not None:
            if self.weight is not None:
                raise RuntimeError(
                    "External weight is provided while at the same time the"
                    " module has defined its own weight parameter. Please"
                    " create the module with flag weight=False."
                )
        else:
            weight = self.weight

        if self._input_size > self._output_size:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = torch.matmul(feat, weight)
            graph.srcdata["h"] = feat
            if edge_weight is None:
                graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
            else:
                graph.edata["edge_weight"] = edge_weight
                graph.update_all(fn.u_mul_e("h", "edge_weight", "m"), fn.sum("m", "h"))
            rst = graph.dstdata["h"]
        else:
            # aggregate first then mult W
            graph.srcdata["h"] = feat
            if edge_weight is None:
                graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
            else:
                graph.edata["edge_weight"] = edge_weight
                graph.update_all(fn.u_mul_e("h", "edge_weight", "m"), fn.sum("m", "h"))
            rst = graph.dstdata["h"]
            if weight is not None:
                rst = torch.matmul(rst, weight)

        if self._gcn_norm != "none":
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._gcn_norm == "both":
                gcn_norm = torch.pow(degs, -0.5)
            else:
                gcn_norm = 1.0 / degs
            shp = gcn_norm.shape + (1,) * (feat.dim() - 1)
            gcn_norm = torch.reshape(gcn_norm, shp)
            rst = rst * gcn_norm

        if self.bias is not None:
            rst = rst + self.bias

        if self.res_fc is not None:
            h_dst = feat_origin
            resval = self.res_fc(h_dst).view(h_dst.shape[0], self._output_size)
            rst = rst + resval

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_input_size}, out={_output_size}"
        summary += ", gcn_normalization={_gcn_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class BiFuseGCNLayerConv(GNNLayerBase):
    r"""Bidirection version GCN layer from paper `GCN <https://arxiv.org/abs/1609.02907>`__.

    .. math::
        h_{i, \vdash}^{(l+1)} = \sigma(b^{(l)}_{\vdash} + \sum_{j\in\mathcal{N}_{\vdash}(i)}
        \frac{1}{c_{ij}}h_{j}^{(l)}W^{(l)}_{\vdash})

        h_{i, \dashv}^{(l+1)} = \sigma(b^{(l)}_{\dashv} + \sum_{j\in\mathcal{N}_{\dashv}(i)}
        \frac{1}{c_{ij}}h_{j}^{(l)}W^{(l)}_{\dashv})

        r_{i}^{l} &= \sigma (W_{f}[h_{i, \vdash}^{l};h_{i, \dashv}^{l};
                h_{i, \vdash}^{l}*h_{i, \dashv}^{l};
                h_{i, \vdash}^{l}-h_{i, \dashv}^{l}])

    Parameters
    ----------
    input_size : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    output_size : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    gcn_norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.
    """

    def __init__(
        self,
        input_size,
        output_size,
        feat_drop=0.0,
        gcn_norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        residual=True,
    ):
        super(BiFuseGCNLayerConv, self).__init__()
        if gcn_norm not in ("none", "both", "right"):
            raise RuntimeError(
                'Invalid gcn_norm value. Must be either "none", "both" or "right".'
                ' But got "{}".'.format(gcn_norm)
            )
        self._input_size = input_size
        self._output_size = output_size
        self._gcn_norm = gcn_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._feat_drop = nn.Dropout(feat_drop)

        if weight:
            self.weight_fw = nn.Parameter(torch.Tensor(input_size, output_size))
            self.weight_bw = nn.Parameter(torch.Tensor(input_size, output_size))
        else:
            self.register_parameter("weight_fw", None)
            self.register_parameter("weight_bw", None)

        if bias:
            self.bias_fw = nn.Parameter(torch.Tensor(output_size))
            self.bias_bw = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter("bias_fw", None)
            self.register_parameter("bias_bw", None)

        self.reset_parameters()

        self._activation = activation

        self.fuse_linear = nn.Linear(4 * output_size, output_size, bias=True)

        if residual:
            if self._input_size != output_size:
                self.res_fc = nn.Linear(self._input_size, output_size, bias=True)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

    def reset_parameters(self):
        r"""
        Reinitialize learnable parameters.
        """
        if self.weight_fw is not None:
            init.xavier_uniform_(self.weight_fw)
            init.xavier_uniform_(self.weight_bw)
        if self.bias_fw is not None:
            init.zeros_(self.bias_fw)
            init.zeros_(self.bias_bw)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None, reverse_edge_weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph: DGLGraph
            The graph.

        feat: torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.

        weight: torch.Tensor, optional
            Optional external weight tensor.

        edge_weight: torch.Tensor
            Optional edge weight. edge_weight shape: "math:`(\text{num_edge}, 1)`.

        reverse_edge_weight: torch.Tensor
            Optional reverse edge weight. reverse_edge_weight shape: "math:`(\text{num_edge}, 1)`.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        feat_fw = feat_bw = self._feat_drop(feat)
        if isinstance(weight, tuple):
            weight_fw, weight_bw = weight
        else:
            weight_fw = weight_bw = weight

        # forward direction
        with graph.local_scope():
            graph = graph.local_var()

            if self._gcn_norm == "both":
                degs = graph.out_degrees().to(feat_fw.device).float().clamp(min=1)
                gcn_norm = torch.pow(degs, -0.5)
                shp = gcn_norm.shape + (1,) * (feat_fw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                feat_fw = feat_fw * gcn_norm

            if weight_fw is not None:
                if self.weight_fw is not None:
                    raise RuntimeError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight_fw = self.weight_fw

            if self._input_size > self._output_size:
                # mult W first to reduce the feature size for aggregation.
                if weight_fw is not None:
                    feat_fw = torch.matmul(feat_fw, weight_fw)
                graph.srcdata["h"] = feat_fw
                if edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["edge_weight"] = edge_weight
                    graph.update_all(fn.u_mul_e("h", "edge_weight", "m"), fn.sum("m", "h"))
                rst_fw = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_fw
                if edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["edge_weight"] = edge_weight
                    graph.update_all(fn.u_mul_e("h", "edge_weight", "m"), fn.sum("m", "h"))
                rst_fw = graph.dstdata["h"]
                if weight_fw is not None:
                    rst_fw = torch.matmul(rst_fw, weight_fw)

            if self._gcn_norm != "none":
                degs = graph.in_degrees().to(feat_fw.device).float().clamp(min=1)
                if self._gcn_norm == "both":
                    gcn_norm = torch.pow(degs, -0.5)
                else:
                    gcn_norm = 1.0 / degs
                shp = gcn_norm.shape + (1,) * (feat_fw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                rst_fw = rst_fw * gcn_norm

            if self.bias_fw is not None:
                rst_fw = rst_fw + self.bias_fw

            if self._activation is not None:
                rst_fw = self._activation(rst_fw)

        # backward direction
        graph = graph.reverse()
        with graph.local_scope():
            graph = graph.local_var()

            if self._gcn_norm == "both":
                degs = graph.out_degrees().to(feat_bw.device).float().clamp(min=1)
                gcn_norm = torch.pow(degs, -0.5)
                shp = gcn_norm.shape + (1,) * (feat_bw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                feat_bw = feat_bw * gcn_norm

            if weight_bw is not None:
                if self.weight_bw is not None:
                    raise RuntimeError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight_bw = self.weight_bw

            if self._input_size > self._output_size:
                # mult W first to reduce the feature size for aggregation.
                if weight_bw is not None:
                    feat_bw = torch.matmul(feat_bw, weight_bw)
                graph.srcdata["h"] = feat_bw
                if reverse_edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["reverse_edge_weight"] = reverse_edge_weight
                    graph.update_all(fn.u_mul_e("h", "reverse_edge_weight", "m"), fn.sum("m", "h"))
                rst_bw = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_bw
                if edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["reverse_edge_weight"] = reverse_edge_weight
                    graph.update_all(fn.u_mul_e("h", "reverse_edge_weight", "m"), fn.sum("m", "h"))
                rst_bw = graph.dstdata["h"]
                if weight_bw is not None:
                    rst_bw = torch.matmul(rst_bw, weight_bw)

            if self._gcn_norm != "none":
                degs = graph.in_degrees().to(feat_bw.device).float().clamp(min=1)
                if self._gcn_norm == "both":
                    gcn_norm = torch.pow(degs, -0.5)
                else:
                    gcn_norm = 1.0 / degs
                shp = gcn_norm.shape + (1,) * (feat_bw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                rst_bw = rst_bw * gcn_norm

            if self.bias_bw is not None:
                rst_bw = rst_bw + self.bias_bw

            if self._activation is not None:
                rst_bw = self._activation(rst_bw)

        fuse_vector = torch.cat([rst_fw, rst_bw, rst_fw * rst_bw, rst_fw - rst_bw], dim=-1)
        fuse_gate_vector = torch.sigmoid(self.fuse_linear(fuse_vector))
        rst = fuse_gate_vector * rst_fw + (1 - fuse_gate_vector) * rst_bw

        if self.res_fc is not None:
            h_dst = feat
            resval = self.res_fc(h_dst).view(h_dst.shape[0], self._output_size)
            rst = rst + resval

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class BiSepGCNLayerConv(GNNLayerBase):
    r"""Bidirection version GCN layer from paper `GCN <https://arxiv.org/abs/1609.02907>`__.

    .. math::
        h_{i, \vdash}^{(l+1)} = \sigma(b^{(l)}_{\vdash} + \sum_{j\in\mathcal{N}_{\vdash}(i)}
        \frac{1}{c_{ij}}h_{j, \vdash}^{(l)}W^{(l)}_{\vdash})

        h_{i, \dashv}^{(l+1)} = \sigma(b^{(l)}_{\dashv} + \sum_{j\in\mathcal{N}_{\dashv}(i)}
        \frac{1}{c_{ij}}h_{j, \dashv}^{(l)}W^{(l)}_{\dashv})
    """

    def __init__(
        self,
        input_size,
        output_size,
        feat_drop=0.0,
        gcn_norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        residual=True,
    ):
        super(BiSepGCNLayerConv, self).__init__()
        if gcn_norm not in ("none", "both", "right"):
            raise RuntimeError(
                'Invalid gcn_norm value. Must be either "none", "both" or "right".'
                ' But got "{}".'.format(gcn_norm)
            )
        self._input_size = input_size
        self._output_size = output_size
        self._gcn_norm = gcn_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._feat_drop = nn.Dropout(feat_drop)

        if weight:
            self.weight_fw = nn.Parameter(torch.Tensor(input_size, output_size))
            self.weight_bw = nn.Parameter(torch.Tensor(input_size, output_size))
        else:
            self.register_parameter("weight_fw", None)
            self.register_parameter("weight_bw", None)

        if bias:
            self.bias_fw = nn.Parameter(torch.Tensor(output_size))
            self.bias_bw = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter("bias_fw", None)
            self.register_parameter("bias_bw", None)

        if residual:
            if self._input_size != output_size:
                self.res_fc_fw = nn.Linear(self._input_size, output_size, bias=True)
                self.res_fc_bw = nn.Linear(self._input_size, output_size, bias=True)
            else:
                self.res_fc_fw = self.res_fc_bw = nn.Identity()
        else:
            self.register_buffer("res_fc_fw", None)
            self.register_buffer("res_fc_bw", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""
        Reinitialize learnable parameters.
        """
        if self.weight_fw is not None:
            init.xavier_uniform_(self.weight_fw)
            init.xavier_uniform_(self.weight_bw)
        if self.bias_fw is not None:
            init.zeros_(self.bias_fw)
            init.zeros_(self.bias_bw)

        if isinstance(self.res_fc_fw, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_fw.weight)

        if isinstance(self.res_fc_bw, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_bw.weight)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None, reverse_edge_weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph: DGLGraph
            The graph.

        feat: torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.

        weight: torch.Tensor, optional
            Optional external weight tensor.

        edge_weight: torch.Tensor
            Optional edge weight. edge_weight shape: "math:`(\text{num_edge}, 1)`.

        reverse_edge_weight: torch.Tensor
            Optional reverse edge weight. reverse_edge_weight shape: "math:`(\text{num_edge}, 1)`.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        feat_fw, feat_bw = feat
        feat_fw = self._feat_drop(feat_fw)
        feat_bw = self._feat_drop(feat_bw)
        if isinstance(weight, tuple):
            weight_fw, weight_bw = weight
        else:
            weight_fw = weight_bw = weight

        # forward direction
        with graph.local_scope():
            graph = graph.local_var()

            if self._gcn_norm == "both":
                degs = graph.out_degrees().to(feat_fw.device).float().clamp(min=1)
                gcn_norm = torch.pow(degs, -0.5)
                shp = gcn_norm.shape + (1,) * (feat_fw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                feat_fw = feat_fw * gcn_norm

            if weight_fw is not None:
                if self.weight_fw is not None:
                    raise RuntimeError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight_fw = self.weight_fw

            if self._input_size > self._output_size:
                # mult W first to reduce the feature size for aggregation.
                if weight_fw is not None:
                    feat_fw = torch.matmul(feat_fw, weight_fw)
                graph.srcdata["h"] = feat_fw
                if edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["edge_weight"] = edge_weight
                    graph.update_all(fn.u_mul_e("h", "edge_weight", "m"), fn.sum("m", "h"))
                rst_fw = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_fw
                if edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["edge_weight"] = edge_weight
                    graph.update_all(fn.u_mul_e("h", "edge_weight", "m"), fn.sum("m", "h"))
                rst_fw = graph.dstdata["h"]
                if weight_fw is not None:
                    rst_fw = torch.matmul(rst_fw, weight_fw)

            if self._gcn_norm != "none":
                degs = graph.in_degrees().to(feat_fw.device).float().clamp(min=1)
                if self._gcn_norm == "both":
                    gcn_norm = torch.pow(degs, -0.5)
                else:
                    gcn_norm = 1.0 / degs
                shp = gcn_norm.shape + (1,) * (feat_fw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                rst_fw = rst_fw * gcn_norm

            if self.bias_fw is not None:
                rst_fw = rst_fw + self.bias_fw

            # residual
            if self.res_fc_fw is not None:
                h_dst_fw = feat[0]
                resval_fw = self.res_fc_fw(h_dst_fw).view(h_dst_fw.shape[0], self._output_size)
                # .view(h_dst.shape[0], self._output_size)
                rst_fw = rst_fw + resval_fw

            if self._activation is not None:
                rst_fw = self._activation(rst_fw)

        # backward direction
        graph = graph.reverse()
        with graph.local_scope():
            graph = graph.local_var()

            if self._gcn_norm == "both":
                degs = graph.out_degrees().to(feat_bw.device).float().clamp(min=1)
                gcn_norm = torch.pow(degs, -0.5)
                shp = gcn_norm.shape + (1,) * (feat_bw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                feat_bw = feat_bw * gcn_norm

            if weight_bw is not None:
                if self.weight_bw is not None:
                    raise RuntimeError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight_bw = self.weight_bw

            if self._input_size > self._output_size:
                # mult W first to reduce the feature size for aggregation.
                if weight_bw is not None:
                    feat_bw = torch.matmul(feat_bw, weight_bw)
                graph.srcdata["h"] = feat_bw
                if reverse_edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["reverse_edge_weight"] = reverse_edge_weight
                    graph.update_all(fn.u_mul_e("h", "reverse_edge_weight", "m"), fn.sum("m", "h"))
                rst_bw = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_bw
                if reverse_edge_weight is None:
                    graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                else:
                    graph.edata["reverse_edge_weight"] = edge_weight
                    graph.update_all(fn.u_mul_e("h", "reverse_edge_weight", "m"), fn.sum("m", "h"))
                rst_bw = graph.dstdata["h"]
                if weight_bw is not None:
                    rst_bw = torch.matmul(rst_bw, weight_bw)

            if self._gcn_norm != "none":
                degs = graph.in_degrees().to(feat_bw.device).float().clamp(min=1)
                if self._gcn_norm == "both":
                    gcn_norm = torch.pow(degs, -0.5)
                else:
                    gcn_norm = 1.0 / degs
                shp = gcn_norm.shape + (1,) * (feat_bw.dim() - 1)
                gcn_norm = torch.reshape(gcn_norm, shp)
                rst_bw = rst_bw * gcn_norm

            if self.bias_bw is not None:
                rst_bw = rst_bw + self.bias_bw

            # residual
            if self.res_fc_bw is not None:
                h_dst_bw = feat[1]
                resval_bw = self.res_fc_bw(h_dst_bw).view(h_dst_bw.shape[0], self._output_size)
                rst_bw = rst_bw + resval_bw

            if self._activation is not None:
                rst_bw = self._activation(rst_bw)

        return [rst_fw, rst_bw]
