import torch
import torch.nn as nn
from torch.nn import init
from dgl.nn import GatedGraphConv
import dgl.function as fn

from .base import GNNLayerBase, GNNBase

class GatedGraphConvWoGRU(nn.Module):
   r"""Gated Graph Convolution layer from paper `Gated Graph Sequence
   Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

   .. math::
       h_{i}^{0} & = [ x_i \| \mathbf{0} ]

       a_{i}^{t} & = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

       h_{i}^{t+1} & = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

   Parameters
   ----------
   in_feats : int
       Input feature size.
   out_feats : int
       Output feature size.
   n_steps : int
       Number of recurrent steps.
   n_etypes : int
       Number of edge types.
   bias : bool
       If True, adds a learnable bias to the output. Default: ``True``.
   """
   def __init__(self,
                in_feats,
                out_feats,
                n_steps,
                n_etypes,
                bias=True):
       super(GatedGraphConvWoGRU, self).__init__()
       self._in_feats = in_feats
       self._out_feats = out_feats
       self._n_steps = n_steps
       self._n_etypes = n_etypes
       self.linears = nn.ModuleList(
           [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
       )
       # self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
       self.reset_parameters()

   def reset_parameters(self):
       """Reinitialize learnable parameters."""
       gain = init.calculate_gain('relu')
       # self.gru.reset_parameters()
       for linear in self.linears:
           init.xavier_normal_(linear.weight, gain=gain)
           init.zeros_(linear.bias)

   def forward(self, graph, feat, etypes):
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

       Returns
       -------
       torch.Tensor
           The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
           is the output feature size.
       """
       assert graph.is_homograph(), \
           "not a homograph; convert it with to_homo and pass in the edge type as argument"
       graph = graph.local_var()
       zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
       feat = torch.cat([feat, zero_pad], -1)

       for _ in range(self._n_steps):
           graph.ndata['h'] = feat
           for i in range(self._n_etypes):
               eids = (etypes == i).nonzero().view(-1)
               if len(eids) > 0:
                   graph.apply_edges(
                       lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                       eids
                   )
           graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
           a = graph.ndata.pop('a')  # (N, D)
           """
           You can also remove node or edge states from the graph. 
           This is particularly useful to save memory during inference.
           """
           # feat = self.gru(a, feat)
       return a


class UniGGNNLayerConv(GNNLayerBase):
    def __init__(self, in_feats, out_feats, n_steps=1, n_etypes=1, bias=True):
        super(UniGGNNLayerConv, self).__init__(in_feats, out_feats)
        self.model = GatedGraphConv(in_feats, out_feats, n_steps=n_steps, n_etypes=n_etypes, bias=bias)

    def forward(self, dgl_graph, node_feats):
        etypes = torch.LongTensor([0]*dgl_graph.number_of_edges())  # [B, E]. E is the number of edges.
        return self.model(dgl_graph, node_feats, etypes)


class BiSepGGNNLayerConv(GNNLayerBase):
    def __init__(self, in_feats, out_feats, bias=True):
        super(BiSepGGNNLayerConv, self).__init__(in_feats, out_feats)
        self.conv_in = GatedGraphConv(in_feats, out_feats, n_steps=1, n_etypes=1, bias=bias)
        self.conv_out = GatedGraphConv(in_feats, out_feats, n_steps=1, n_etypes=1, bias=bias)
        self.forward_linear = nn.Linear(in_feats + out_feats, out_feats, bias=True)
        self.backward_linear = nn.Linear(in_feats + out_feats, out_feats, bias=True)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'forward_linear'):
            nn.init.xavier_normal_(self.forward_linear.weight, gain=gain)

        if hasattr(self, 'backward_linear'):
            nn.init.xavier_normal_(self.backward_linear.weight, gain=gain)

    def forward(self, graph, node_feats):
        feat_in, feat_out = node_feats
        etypes = torch.LongTensor([0]*graph.number_of_edges())  # [B, E]. E is the number of edges.

        forward_graph = graph
        emb_in = self.conv_in(forward_graph, feat_in, etypes)

        backward_graph = graph.reverse()
        emb_out = self.conv_out(backward_graph, feat_out, etypes)

        concat_in = torch.cat([feat_in, emb_in], dim=-1)
        rst_in = torch.sigmoid(self.forward_linear(concat_in))

        concat_out = torch.cat([feat_out, emb_out], dim=-1)
        rst_out = torch.sigmoid(self.backward_linear(concat_out))


        return [rst_in, rst_out]


class BiFuseGGNNLayerConv(GNNLayerBase):
    def __init__(self, in_feats, out_feats, bias=True):
        super(BiFuseGGNNLayerConv, self).__init__(in_feats, out_feats)
        self.conv_in = GatedGraphConvWoGRU(in_feats, out_feats, n_steps=1, n_etypes=1)
        self.conv_out = GatedGraphConvWoGRU(in_feats, out_feats, n_steps=1, n_etypes=1)
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.fuse_linear = nn.Linear(4 * out_feats, out_feats, bias=True)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        self.gru.reset_parameters()
        if hasattr(self, 'fuse_linear'):
            nn.init.xavier_normal_(self.fuse_linear.weight, gain=gain)

    def forward(self, graphs, node_feats):
        feat_in, feat_out = node_feats  # feat_in == feat_out
        etypes = torch.LongTensor([0]*graphs.number_of_edges())  # [B, E]. E is the number of edges.

        # forward aggregation
        forward_graph = graphs
        emb_in = self.conv_in(forward_graph, feat_in, etypes)

        # backward aggregation
        backward_graph = graphs.reverse()
        emb_out = self.conv_out(backward_graph, feat_out, etypes)

        # fuse
        fuse_vector = torch.cat(
            [emb_in, emb_out, emb_in * emb_out, emb_in - emb_out], dim=-1)
        fuse_gate_vector = torch.sigmoid(self.fuse_linear(fuse_vector))
        emb_fused = fuse_gate_vector * emb_out + (1 - fuse_gate_vector) * emb_out

        # update
        rst = self.gru(emb_fused, feat_in)

        return [rst, rst]



class GGNNLayer(GNNLayerBase):
    def __init__(self, in_feats, out_feats, direction_option='uni', n_steps=1, n_etypes=1, bias=True):
        super(GGNNLayer, self).__init__(in_feats, out_feats)
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
    def __init__(self, num_layers, in_feats, out_feats, direction_option='uni', n_etypes=1, bias=True):
        r"""
        Gated Graph Neural Networks from paper `Gated Graph Sequence
        Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

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
        super(GGNN, self).__init__(in_feats, out_feats)
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
