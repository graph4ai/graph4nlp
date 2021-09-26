import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from torch.nn import init

from ...data.data import GraphData
from .base import GNNBase, GNNLayerBase


class UndirectedGGNNLayerConv(GNNLayerBase):
    r"""
    Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
       h_{i}^{0} = [ x_i \| \mathbf{0} ]

       a_{i}^{t} = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

       h_{i}^{t+1} = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    input_size: int
        Input feature size.
    output_size: int
        Output feature size.
    num_layers: int
        Number of GGNN layers. Default: 1.
    n_etypes: int
        Number of edge types. Default: 1.
    bias: bool
        If True, adds a learnable bias to the output. Default: True.
    """

    def __init__(self, input_size, output_size, num_layers, n_etypes, bias=True):
        super(UndirectedGGNNLayerConv, self).__init__()
        self._in_feats = input_size
        self._out_feats = output_size
        self._num_layers = num_layers
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList([nn.Linear(output_size, output_size) for _ in range(n_etypes)])
        self.gru = nn.GRUCell(output_size, output_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain("relu")
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def forward(self, graph, feat, etypes=None, edge_weight=None):
        """Compute Gated Graph Convolution layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph.
        edge_weight: torch.Tensor
            The shape of edge_weight is :math:`(N_E, 1)`. N_E is the number of edges in graph.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is size of output feature.
        """
        if dgl.__version__ < "0.5":
            assert (
                graph.is_homograph()
            ), "not a homograph; convert it with to_homo and pass in the edge type as argument"
        else:
            assert (
                graph.is_homogeneous
            ), "not a homograph; convert it with to_homo and pass in the edge type as argument"
        graph = graph.local_var()
        zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
        feat = torch.cat([feat, zero_pad], -1)

        for _ in range(self._num_layers):
            graph.ndata["h"] = feat
            for i in range(self._n_etypes):
                eids = (etypes == i).nonzero(as_tuple=False).view(-1)
                if len(eids) > 0:
                    if edge_weight is None:
                        graph.apply_edges(
                            lambda edges: {"W_e*h": self.linears[i](edges.src["h"])}, eids
                        )
                    else:
                        graph.apply_edges(
                            lambda edges: {
                                "W_e*h": self.linears[i](edges.src["h"])
                                * edge_weight.view(-1).unsqueeze(1)
                            },
                            eids,
                        )
            graph.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
            a = graph.ndata.pop("a")  # (N, D)
            feat = self.gru(a, feat)
        return feat


class BiFuseGGNNLayerConv(GNNLayerBase):
    r"""
    Fuse aggregated embeddings from both incoming and outgoing
    directions before updating node embeddings.
    .. math::
       h_{i}^{0} = [ x_i \| \mathbf{0} ]
       a_{i, \vdash}^{t} = \sum_{j\in\mathcal{N}_{\vdash }(i)}
       W_{\vdash} h_{j}^{t}
       a_{i, \dashv}^{t} = \sum_{j\in\mathcal{N}_{\dashv }(i)}
       W_{\dashv} h_{j}^{t}
       e_{i}^{t} &= \sigma (W_{f}[a_{i, \vdash}^{t};a_{i, \dashv}^{t};
       a_{i, \vdash}^{t}*a_{i, \dashv}^{t};
       a_{i, \vdash}^{t}-a_{i, \dashv}^{t}])
       h_{i}^{t+1} = \mathrm{GRU}(e_{i}^{t}, h_{i}^{t})
    Attributes
    ----------
    input_size: int
        Input feature size.
    output_size: int
        Output feature size.
    n_etypes: int
        Number of edge types. Default: 1.
    bias: bool
        If True, adds a learnable bias to the output. Default: True.
    Attributes
    ----------
    input_size: int
        Input feature size.
    output_size: int
        Output feature size.
    n_etypes: int
        Number of edge types. Default: 1.
    bias: bool
        If True, adds a learnable bias to the output. Default: True.
    """

    def __init__(self, input_size, output_size, n_etypes=1, bias=True):
        super(BiFuseGGNNLayerConv, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        # self._num_layers = num_layers
        self._n_etypes = n_etypes

        self.linears_in = nn.ModuleList(
            [nn.Linear(output_size, output_size) for _ in range(n_etypes)]
        )

        self.linears_out = nn.ModuleList(
            [nn.Linear(output_size, output_size) for _ in range(n_etypes)]
        )

        self.gru = nn.GRUCell(output_size, output_size, bias=bias)
        self.fuse_linear = nn.Linear(4 * output_size, output_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        self.gru.reset_parameters()
        nn.init.xavier_normal_(self.fuse_linear.weight, gain=gain)

        for linear in self.linears_in:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

        for linear in self.linears_out:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

    def forward(self, graph, node_feats, etypes=None, edge_weight=None):
        """
        Parameters
        ----------
        graph: dgl.DGLGraph
        node_feats: torch.Tensor
            The shape of node_feats is :math:`(N, D_{in})`.
        edge_weight: torch.Tensor
            The shape of edge_weight is :math:`(N_E, 1)`. N_E is the number of edges in graph.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is size of output feature.
        """
        feat_in, feat_out = node_feats  # feat_in == feat_out

        # forward aggregation
        graph_in = graph
        graph_in = graph_in.local_var()
        graph_in.ndata["h"] = feat_in
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero(as_tuple=False).view(-1)
            if len(eids) > 0:
                if edge_weight is None:
                    graph_in.apply_edges(
                        lambda edges: {"W_e*h": self.linears_in[i](edges.src["h"])}, eids
                    )
                else:
                    assert isinstance(edge_weight, tuple)
                    graph_in.apply_edges(
                        lambda edges: {
                            "W_e*h": self.linears_in[i](edges.src["h"])
                            * edge_weight[0].view(-1).unsqueeze(1)
                        },
                        eids,
                    )
        graph_in.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
        agg_in = graph_in.ndata.pop("a")  # (N, D)

        # backward aggregation
        graph_out = graph.reverse()
        graph_out = graph_out.local_var()
        graph_out.ndata["h"] = feat_out
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero(as_tuple=False).view(-1)
            if len(eids) > 0:
                if edge_weight is None:
                    graph_out.apply_edges(
                        lambda edges: {"W_e*h": self.linears_out[i](edges.src["h"])}, eids
                    )
                else:
                    assert isinstance(edge_weight, tuple)
                    graph_out.apply_edges(
                        lambda edges: {
                            "W_e*h": self.linears_out[i](edges.src["h"])
                            * edge_weight[1].view(-1).unsqueeze(1)
                        },
                        eids,
                    )
        graph_out.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
        agg_out = graph_out.ndata.pop("a")  # (N, D)

        # fuse
        fuse_vector = torch.cat([agg_in, agg_out, agg_in * agg_out, agg_in - agg_out], dim=-1)
        fuse_gate_vector = torch.sigmoid(self.fuse_linear(fuse_vector))
        emb_fused = fuse_gate_vector * agg_in + (1 - fuse_gate_vector) * agg_out

        # update
        rst = self.gru(emb_fused, feat_in)

        return [rst, rst]


class BiSepGGNNLayerConv(GNNLayerBase):
    r"""
    Compute node embeddings for incoming and outgoing directions
    separately, and then concatenate the two output node embeddings
    after the final layer.
    .. math::
       h_{i}^{0} = [ x_i \| \mathbf{0} ]

       a_{i, \vdash}^{t} = \sum_{j\in\mathcal{N}_{\vdash }(i)} W_{\vdash} h_{j, \vdash}^{t}

       a_{i, \dashv}^{t} = \sum_{j\in\mathcal{N}_{\dashv }(i)} W_{\dashv} h_{j, \dashv}^{t}

       h_{i, \vdash}^{t+1} = \mathrm{GRU}_{\vdash}(a_{i, \vdash}^{t}, h_{i, \vdash}^{t})

       h_{i, \dashv}^{t+1} = \mathrm{GRU}_{\dashv}(a_{i, \dashv}^{t}, h_{i, \dashv}^{t})
    Attributes
    ----------
    input_size: int
        Input feature size.
    output_size: int
        Output feature size.
    n_etypes: int
        Number of edge types. Default: 1.
    bias: bool
        If True, adds a learnable bias to the output. Default: True.
    Attributes
    ----------
    input_size: int
        Input feature size.
    output_size: int
        Output feature size.
    n_etypes: int
        Number of edge types. Default: 1.
    bias: bool
        If True, adds a learnable bias to the output. Default: True.
    """

    def __init__(self, input_size, output_size, n_etypes=1, bias=True):
        super(BiSepGGNNLayerConv, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        # self._num_layers = num_layers
        self._n_etypes = n_etypes

        self.linears_in = nn.ModuleList(
            [nn.Linear(output_size, output_size) for _ in range(n_etypes)]
        )

        self.linears_out = nn.ModuleList(
            [nn.Linear(output_size, output_size) for _ in range(n_etypes)]
        )

        # self.update_in = nn.Linear(input_size + output_size, output_size, bias=True)
        # self.update_out = nn.Linear(input_size + output_size, output_size, bias=True)

        self.gru_in = nn.GRUCell(output_size, output_size, bias=bias)
        self.gru_out = nn.GRUCell(output_size, output_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        self.gru_in.reset_parameters()
        self.gru_out.reset_parameters()

        for linear in self.linears_in:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

        for linear in self.linears_out:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

        # nn.init.xavier_normal_(self.update_in.weight, gain=gain)
        # nn.init.xavier_normal_(self.update_out.weight, gain=gain)

    def forward(self, graph, node_feats, etypes=None, edge_weight=None):
        feat_in, feat_out = node_feats

        graph_in = graph
        graph_in = graph_in.local_var()
        graph_in.ndata["h"] = feat_in
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero(as_tuple=False).view(-1)
            if len(eids) > 0:
                if edge_weight is None:
                    graph_in.apply_edges(
                        lambda edges: {"W_e*h": self.linears_in[i](edges.src["h"])}, eids
                    )
                else:
                    assert isinstance(edge_weight, tuple)
                    graph_in.apply_edges(
                        lambda edges: {
                            "W_e*h": self.linears_in[i](edges.src["h"])
                            * edge_weight[0].view(-1).unsqueeze(1)
                        },
                        eids,
                    )
        graph_in.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
        a_in = graph_in.ndata.pop("a")  # (N, D)
        emb_in = self.gru_in(a_in, feat_in)

        graph_out = graph.reverse()
        graph_out = graph_out.local_var()
        graph_out.ndata["h"] = feat_out
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero(as_tuple=False).view(-1)
            if len(eids) > 0:
                if edge_weight is None:
                    graph_out.apply_edges(
                        lambda edges: {"W_e*h": self.linears_out[i](edges.src["h"])}, eids
                    )
                else:
                    assert isinstance(edge_weight, tuple)
                    graph_out.apply_edges(
                        lambda edges: {
                            "W_e*h": self.linears_out[i](edges.src["h"])
                            * edge_weight[1].view(-1).unsqueeze(1)
                        },
                        eids,
                    )
        graph_out.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
        a_out = graph_out.ndata.pop("a")  # (N, D)
        emb_out = self.gru_out(a_out, feat_out)

        return [emb_in, emb_out]


class GGNNLayer(GNNLayerBase):
    r"""A unified wrapper for Gated Graph Convolution layer
    from paper `Gated Graph Sequence Neural Networks
    <https://arxiv.org/pdf/1511.05493.pdf>`__.
    Support both undirected (i.e., regular) and bidirectional
    (i.e., `bi_sep` and `bi_fuse`) versions.
    Attributes
    ----------
    input_size: int
        Input feature size.
    output_size: int
        Output feature size.
    direction_option: str
        The direction option of GGNN ('undirected', 'bi_sep' or 'bi_fuse'). (Default: 'bi_fuse')
    num_layers: int
        Number of GGNN layers.
        `num_layers` is set to any integer if the direction_option is 'undirected'.
        If the direction_option is 'bi_sep' or 'bi_fuse', `num_layers` will be set to 1.
    n_etypes: int
        Number of edge types. `n_etypes` can be set to any integer if the
        direction_option is 'undirected'.
        If the direction_option is 'bi_sep' or 'bi_fuse', `n_etypes` will be set to 1.
    bias: bool
        If True, adds a learnable bias to the output. (Default: True)
    """

    def __init__(
        self,
        input_size,
        output_size,
        direction_option="bi_fuse",
        num_layers=1,
        n_etypes=1,
        bias=True,
    ):
        super(GGNNLayer, self).__init__()
        if direction_option == "undirected":
            self.model = UndirectedGGNNLayerConv(
                input_size, output_size, num_layers=num_layers, n_etypes=n_etypes, bias=bias
            )
        elif direction_option == "bi_sep":
            self.model = BiSepGGNNLayerConv(input_size, output_size, n_etypes, bias=bias)
        elif direction_option == "bi_fuse":
            self.model = BiFuseGGNNLayerConv(input_size, output_size, n_etypes, bias=bias)
        else:
            raise RuntimeError("Unknown `bidirection` value: {}".format(direction_option))

    def forward(self, graph, node_feats, etypes=None, edge_weight=None):
        """
        Parameters
        ----------
        graph: dgl.DGLGraph
        node_feats: torch.Tensor
            The shape of node_feats is :math:`(N, D_{in})`.
        Returns
        -------
        torch.Tensor
        """
        if etypes is None:
            etypes = torch.tensor([0] * graph.number_of_edges(), dtype=torch.long).to(graph.device)
        return self.model(graph, node_feats, etypes, edge_weight)


class GGNN(GNNBase):
    r"""
    Multi-layered `Gated Graph Sequence Neural Networks
    <https://arxiv.org/pdf/1511.05493.pdf>`__.
    Support both undirected (i.e., regular) and bidirectional
    (i.e., `bi_sep` and `bi_fuse`) versions.
    Attributes
    ----------
    num_layers: int
        Number of GGNN layers.
    input_size: int
        Input feature size.
    hidden_size: int
        Should be equal to output_size.
    output_size: int
        Output feature size.
    direction_option: str
        The direction option of GGNN ('undirected', 'bi_sep' or 'bi_fuse').
        (Default: 'bi_fuse')
    n_etypes: int
        Number of edge types. n_etypes can be set to any integer if the
        direction_option is 'undirected'.
        If the direction_option is 'bi_sep' or 'bi_fuse', n_etypes will be set to 1.
    bias: bool
        If True, adds a learnable bias to the output. (Default: True)
    """

    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        feat_drop=0.0,
        direction_option="bi_fuse",
        n_etypes=1,
        bias=True,
        use_edge_weight=False,
    ):
        super(GGNN, self).__init__()
        self.num_layers = num_layers
        self.direction_option = direction_option
        self.input_size = input_size
        self.output_size = output_size
        self.feat_drop = nn.Dropout(feat_drop)
        self.use_edge_weight = use_edge_weight
        self.n_etypes = n_etypes

        assert self.output_size >= self.input_size
        assert self.output_size == hidden_size

        if self.direction_option == "undirected":
            self.models = GGNNLayer(
                input_size,
                output_size,
                direction_option,
                num_layers=num_layers,
                n_etypes=n_etypes,
                bias=bias,
            )
        else:
            self.models = GGNNLayer(
                input_size, output_size, direction_option, n_etypes=n_etypes, bias=bias
            )

    def forward(self, graph: GraphData):
        r"""
        Use GGNN compute node embeddings.
        Parameters
        ----------
        graph: GraphData.
            The initial node features are stored in the node feature field
            named `node_feat`.
        Returns
        -------
        input_graph: GraphData.
            The computed node embedding tensors are stored in the node feature field
            named `node_emb`.
        """
        if self.n_etypes == 1:
            graph.edge_features["etype"] = torch.tensor(
                [0] * graph.get_edge_num(), dtype=torch.long, device=graph.device
            )

        node_feats = graph.node_features["node_feat"]
        etypes = graph.edge_features["etype"]
        if self.use_edge_weight:
            edge_weight = graph.edge_features["edge_weight"]
            if self.direction_option == "bi_fuse" or self.direction_option == "bi_sep":
                reverse_edge_weight = graph.edge_features["reverse_edge_weight"]
                edge_weight = (edge_weight, reverse_edge_weight)
        else:
            edge_weight = None

        dgl_graph = graph.to_dgl()

        if self.direction_option == "undirected":
            node_embs = self.models(dgl_graph, node_feats, etypes, edge_weight)
        else:
            assert node_feats.shape[1] == self.input_size

            zero_pad = node_feats.new_zeros(
                (node_feats.shape[0], self.output_size - node_feats.shape[1])
            )
            node_feats = torch.cat([node_feats, zero_pad], -1)

            feat_in = node_feats
            feat_out = node_feats

            for _ in range(self.num_layers):
                feat_in = self.feat_drop(feat_in)
                feat_out = self.feat_drop(feat_out)
                h = self.models(dgl_graph, (feat_in, feat_out), etypes, edge_weight)
                feat_in = h[0]
                feat_out = h[1]

            if self.direction_option == "bi_sep":
                node_embs = torch.cat([feat_in, feat_out], dim=-1)
            elif self.direction_option == "bi_fuse":
                node_embs = feat_in
            else:
                raise RuntimeError("Unknown `bidirection` value: {}".format(self.direction_option))

        graph.node_features["node_emb"] = node_embs

        return graph
