import torch
import torch.nn as nn
from dgl.nn import GatedGraphConv
import dgl.function as fn

from .base import GNNLayerBase, GNNBase
from ...data.data import GraphData


class UniGGNNLayerConv(GNNLayerBase):
    r"""
    Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
       h_{i}^{0} & = [ x_i \| \mathbf{0} ]

       a_{i}^{t} & = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

       h_{i}^{t+1} & = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    """
    def __init__(self,
                 input_size,
                 output_size,
                 n_steps=1,
                 n_etypes=1,
                 bias=True):
        """

        Parameters
        ----------
        input_size: int
            Input feature size.
        output_size: int
            Output feature size.
        n_steps: int
            Number of GGNN layers. default: 1.
        n_etypes: int
            Number of edge types. default: 1.
        bias: bool
            If True, adds a learnable bias to the output. default: True.
        """
        super(UniGGNNLayerConv, self).__init__()
        self.model = GatedGraphConv(input_size, output_size, n_steps=n_steps, n_etypes=n_etypes, bias=bias)

    def forward(self, graph, node_feats):
        """

        Parameters
        ----------
        graph: dgl.DGLGraph

        node_feats: torch.Tensor
            The shape of node_feats is :math:`(N, D_{in})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is size of output feature.

        """
        etypes = torch.LongTensor([0] * graph.number_of_edges())  # [B, E]. E is the number of edges.
        return self.model(graph, node_feats, etypes)


class BiFuseGGNNLayerConv(GNNLayerBase):
    r"""
    Fuse aggregated embeddings from both incoming and outgoing
    directions before updating node embeddings.

    .. math::
       h_{i}^{0} & = [ x_i \| \mathbf{0} ]

       a_{i, \vdash}^{t} & = \sum_{j\in\mathcal{N}_{\vdash }(i)} 
       W_{\vdash} h_{j}^{t}

       a_{i, \dashv}^{t} & = \sum_{j\in\mathcal{N}_{\dashv }(i)} 
       W_{\dashv} h_{j}^{t}

       e_{i}^{t} &= \sigma (W_{f}[a_{i, \vdash}^{t};a_{i, \dashv}^{t};
       a_{i, \vdash}^{t}*a_{i, \dashv}^{t};
       a_{i, \vdash}^{t}-a_{i, \dashv}^{t}])

       h_{i}^{t+1} & = \mathrm{GRU}(e_{i}^{t}, h_{i}^{t})

    """
    def __init__(self, in_feats, out_feats, n_etypes=1, bias=True):
        super(BiFuseGGNNLayerConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        # self._n_steps = n_steps
        self._n_etypes = n_etypes

        self.linears_in = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )

        self.linears_out = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )

        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.fuse_linear = nn.Linear(4 * out_feats, out_feats, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        self.gru.reset_parameters()
        nn.init.xavier_normal_(self.fuse_linear.weight, gain=gain)

        for linear in self.linears_in:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

        for linear in self.linears_out:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)


    def forward(self, graph, node_feats):
        """

        Parameters
        ----------
        graph: dgl.DGLGraph

        node_feats: torch.Tensor
            The shape of node_feats is :math:`(N, D_{in})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is size of output feature.

        """
        feat_in, feat_out = node_feats  # feat_in == feat_out
        etypes = torch.LongTensor([0]*graph.number_of_edges())  # [B, E]. E is the number of edges.

        # forward aggregation
        graph_in = graph
        graph_in = graph_in.local_var()
        graph_in.ndata['h'] = feat_in
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero().view(-1)
            if len(eids) > 0:
                graph_in.apply_edges(
                    lambda edges: {'W_e*h': self.linears_in[i](edges.src['h'])},
                    eids
                )
        graph_in.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
        agg_in = graph_in.ndata.pop('a')  # (N, D)

        # backward aggregation
        graph_out = graph.reverse()
        graph_out = graph_out.local_var()
        graph_out.ndata['h'] = feat_out
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero().view(-1)
            if len(eids) > 0:
                graph_out.apply_edges(
                    lambda edges: {'W_e*h': self.linears_out[i](edges.src['h'])},
                    eids
                )
        graph_out.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
        agg_out = graph_out.ndata.pop('a')  # (N, D)

        # fuse
        fuse_vector = torch.cat(
            [agg_in, agg_out, agg_in * agg_out, agg_in - agg_out], dim=-1)
        fuse_gate_vector = torch.sigmoid(self.fuse_linear(fuse_vector))
        emb_fused = fuse_gate_vector * agg_out + (1 - fuse_gate_vector) * agg_out

        # update
        rst = self.gru(emb_fused, feat_in)

        return [rst, rst]


class BiSepGGNNLayerConv(GNNLayerBase):
    r"""
    Compute node embeddings for incoming and outgoing directions
    separately, and then concatenate the two output node embeddings
    after the final layer.

    .. math::
       h_{i}^{0} & = [ x_i \| \mathbf{0} ]

       a_{i, \vdash}^{t} & = \sum_{j\in\mathcal{N}_{\vdash }(i)}
       W_{\vdash} h_{j, \vdash}^{t}

       a_{i, \dashv}^{t} & = \sum_{j\in\mathcal{N}_{\dashv }(i)}
       W_{\dashv} h_{j, \dashv}^{t}

       h_{i, \vdash}^{t+1} & = \mathrm{GRU}_{\vdash}(a_{i, \vdash}^{t},
       h_{i, \vdash}^{t})

       h_{i, \dashv}^{t+1} & = \mathrm{GRU}_{\dashv}(a_{i, \dashv}^{t},
       h_{i, \dashv}^{t})

    """
    def __init__(self, in_feats, out_feats, n_etypes=1, bias=True):
        super(BiSepGGNNLayerConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        # self._n_steps = n_steps
        self._n_etypes = n_etypes

        self.linears_in = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )

        self.linears_out = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )

        self.update_in = nn.Linear(in_feats + out_feats, out_feats, bias=True)
        self.update_out = nn.Linear(in_feats + out_feats, out_feats, bias=True)

        self.gru_in = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.gru_out = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        self.gru_in.reset_parameters()
        self.gru_out.reset_parameters()

        for linear in self.linears_in:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

        for linear in self.linears_out:
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)

        nn.init.xavier_normal_(self.update_in.weight, gain=gain)
        nn.init.xavier_normal_(self.update_out.weight, gain=gain)

    def forward(self, graph, node_feats):
        feat_in, feat_out = node_feats
        etypes = torch.LongTensor([0]*graph.number_of_edges())  # [B, E]. E is the number of edges.

        graph_in = graph
        graph_in = graph_in.local_var()
        graph_in.ndata['h'] = feat_in
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero().view(-1)
            if len(eids) > 0:
                graph_in.apply_edges(
                    lambda edges: {'W_e*h': self.linears_in[i](edges.src['h'])},
                    eids
                )
        graph_in.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
        a_in = graph_in.ndata.pop('a')  # (N, D)
        emb_in = self.gru_in(a_in, feat_in)

        graph_out = graph.reverse()
        graph_out = graph_out.local_var()
        graph_out.ndata['h'] = feat_out
        for i in range(self._n_etypes):
            eids = (etypes == i).nonzero().view(-1)
            if len(eids) > 0:
                graph_out.apply_edges(
                    lambda edges: {'W_e*h': self.linears_out[i](edges.src['h'])},
                    eids
                )
        graph_out.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
        a_out = graph_out.ndata.pop('a')  # (N, D)
        emb_out = self.gru_out(a_out, feat_out)

        concat_in = torch.cat([feat_in, emb_in], dim=-1)
        rst_in = torch.sigmoid(self.update_in(concat_in))

        concat_out = torch.cat([feat_out, emb_out], dim=-1)
        rst_out = torch.sigmoid(self.update_out(concat_out))

        return [rst_in, rst_out]


class GGNNLayer(GNNLayerBase):
    r"""A unified wrapper for Gated Graph Convolution layer
    from paper `Gated Graph Sequence Neural Networks
    <https://arxiv.org/pdf/1511.05493.pdf>`__.

    Support both unidirectional (i.e., regular) and bidirectional
    (i.e., `bi_sep` and `bi_fuse`) versions.
    """
    def __init__(self, in_feats, out_feats, direction_option='uni', n_steps=1, n_etypes=1, bias=True):
        super(GGNNLayer, self).__init__()
        if direction_option == 'uni':
            self.model = UniGGNNLayerConv(in_feats, out_feats, n_steps=n_steps, n_etypes=n_etypes, bias=bias)
        elif direction_option == 'bi_sep':
            self.model = BiSepGGNNLayerConv(in_feats, out_feats, bias=bias)
        elif direction_option == 'bi_fuse':
            self.model = BiFuseGGNNLayerConv(in_feats, out_feats, bias=bias)
        else:
            raise RuntimeError('Unknown `bidirection` value: {}'.format(direction_option))

    def forward(self, graph, node_feats):
        return self.model(graph, node_feats)


class GGNN(GNNBase):
    r"""
    Multi-layered `Gated Graph Sequence Neural Networks
    <https://arxiv.org/pdf/1511.05493.pdf>`__.
    Support both unidirectional (i.e., regular) and bidirectional
    (i.e., `bi_sep` and `bi_fuse`) versions.
    """
    def __init__(self, num_layers, in_feats, out_feats, direction_option='uni', n_etypes=1, bias=True):
        r"""

        Parameters
        ----------
        num_layers: int
            Number of GGNN layers.
        in_feats: int
            Input feature size.
        out_feats: int
            Output feature size.
        direction_option: str
            The direction option of GGNN ('uni', 'bi_sep' or 'bi_fuse'). (default: 'uni')
        n_etypes: int
            Number of edge types. n_etypes can be set to any integer if the direction_option is 'uni'.
            If the direction_option is 'bi_sep' or 'bi_fuse', n_etypes will be set to 1.
        bias: bool
            If True, adds a learnable bias to the output. (default: True)


        """
        super(GGNN, self).__init__()
        self.num_layers = num_layers
        self.direction_option = direction_option
        self.in_feats = in_feats
        self.out_feats = out_feats

        assert self.out_feats >= self.in_feats

        if self.direction_option=='uni':
            self.models = GGNNLayer(in_feats, out_feats, direction_option, n_steps=num_layers, n_etypes=n_etypes, bias=bias)
        else:
            self.models = GGNNLayer(out_feats, out_feats, direction_option, bias=bias)

    def forward(self, graph, node_feats):
        r"""
        Use GGNN compute node embeddings.

        Parameters
        ----------
        graph: dgl.DGLGraph.
            contains N nodes and E edges
        node_feats: torch.Tensor.
            The dimension of node_embs is [N, D_in].

        Returns
        -------
        node_embs: torch.Tensor.
            If the direction_option == 'bi_sep', the dimension of node_embs is [N, 2*D_out].
            Else, the dimension of node_embs is [N, D_out].

        """
        if self.direction_option == 'uni':
            node_embs = self.models(graph, node_feats)
        else:
            assert node_feats.shape[1] == self.in_feats

            zero_pad = node_feats.new_zeros((node_feats.shape[0], self.out_feats - node_feats.shape[1]))
            node_feats = torch.cat([node_feats, zero_pad], -1)

            feat_in = node_feats
            feat_out = node_feats

            for i in range(self.num_layers):
                h = self.models(graph, (feat_in, feat_out))
                feat_in = h[0]
                feat_out = h[1]

            if self.direction_option == 'bi_sep':
                node_embs = torch.cat([feat_in, feat_out], dim=-1)
            elif self.direction_option == 'bi_fuse':
                node_embs = feat_in
            else:
                raise RuntimeError('Unknown `bidirection` value: {}'.format(self.direction_option))

        return node_embs


class HighLevelGGNN(nn.Module):
    r"""
    High-level `Gated Graph Sequence Neural Networks
    <https://arxiv.org/pdf/1511.05493.pdf>`__ receives `GraphData` as input.
    Support both unidirectional (i.e., regular) and bidirectional
    (i.e., `bi_sep` and `bi_fuse`) versions.
    """
    def __init__(self, num_layers, in_feats, out_feats, direction_option='uni', n_etypes=1, bias=True):
        r"""

        Parameters
        ----------
        num_layers: int
            Number of GGNN layers.
        in_feats: int
            Input feature size.
        out_feats: int
            Output feature size.
        direction_option: str
            The direction option of GGNN ('uni', 'bi_sep' or 'bi_fuse'). (default: 'uni')
        n_etypes: int
            Number of edge types. n_etypes can be set to any integer if the direction_option is 'uni'.
            If the direction_option is 'bi_sep' or 'bi_fuse', n_etypes will be set to 1.
        bias: bool
            If True, adds a learnable bias to the output. (default: True)


        """
        super(HighLevelGGNN, self).__init__()

        assert out_feats >= in_feats

        self.models = GGNN(num_layers, in_feats, out_feats, direction_option, n_etypes, bias)

    def forward(self, input_graph):
        r"""
        Use GGNN compute node embeddings.

        Parameters
        ----------
        input_graph: GraphData.
            The initial node features are stored in the node feature field
            named `node_feats`.

        Returns
        -------
        input_graph: GraphData.
            The computed node embedding tensors are stored in the node feature field
            named `node_embs`.

        """

        graph = input_graph.to_dgl()
        node_feats = input_graph._node_features['node_feats']

        input_graph._node_features['node_embs'] = self.models(graph, node_feats)

        return input_graph