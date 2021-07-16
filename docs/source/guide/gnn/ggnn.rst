.. _guide-ggnn:

Gated Graph Neural Networks
==============

A typical example of recurrent-based graph filters is the Gated Graph Neural Networks (`GGNN <https://arxiv.org/pdf/1511.05493.pdf>`_)-filter.
The biggest modification from typical GNNs to GGNNs is the use of Gated Recurrent Units (GRU).
The GGNN-filter also takes the edge type and edge direction into consideration.
To this end, :math:`e_{i,j}` denotes the directed edge from node :math:`v_i` to node :math:`v_j`
and the edge type of :math:`e_{i,j}` is :math:`t_{i,j}`. The propagation process of recurrent-based
filter  :math:`f_\mathbf{filter}` in GGNN can be summarized as follows:

.. math::
    \mathbf{h}_i^{(0)} = [\mathbf{x}_i^T, \mathbf{0}]^T

    \mathbf{a}_i^{(l)} = A_{i:}^T[\mathbf{h}_1^{(l-1)}...\mathbf{h}_n^{(l-1)}]^T

    \mathbf{h}_i^{(l)} = \text{GRU}(\mathbf{a}_i^{(l)}, \mathbf{h}_i^{(l-1)})

where :math:`A \in \mathbb{R}^{{dn} \times 2dn}` is a matrix determining how nodes in the
graph communicating with each other. :math:`n` is the number of nodes in the graph.
:math:`A_{i:} \in \mathbb{R}^{d \times 2d}` are the two columns of blocks in :math:`A`
corresponding to node :math:`v_i`. In Eq. \eqref{ggnn-0}, the initial node feature
:math:`\mathbf{x}_i` are padded with extra zeros to make the input size equal to the
hidden size. Eq. \eqref{eq:ggnn-aggregation} computes
:math:`\mathbf{a}_i^{(l)} \in \mathbb{R}^{2d}` by aggregating information from different
nodes via incoming and outgoing edges with parameters dependent on the edge type
and direction. The following step uses a GRU unit to update the hidden state of
node :math:`v` by incorporating :math:`\mathbf{a}_i^{(l)}` and the previous timestep hidden
state :math:`\mathbf{h}_i^{(l-1)}`.

4.2.1 GGNN Module Construction Function
---------------------------------------

The construction function performs the following steps:

1. Set options.
2. Register learnable parameters or submodules (``GGNNLayer``).

.. code::

    class GGNN(GNNBase):
        def __init__(self, num_layers, input_size, hidden_size, output_size, feat_drop=0.,
                     direction_option='bi_fuse', n_etypes=1, bias=True, use_edge_weight=False):
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

            if self.direction_option == 'undirected':
                self.models = GGNNLayer(input_size, output_size, direction_option, num_layers=num_layers, n_etypes=n_etypes,
                                        bias=bias)
            else:
                self.models = GGNNLayer(input_size, output_size, direction_option, n_etypes=n_etypes, bias=bias)

``hidden_size`` should be equal to output_size.

``n_etypes`` Number of edge types. n_etypes can be set to any integer if the direction_option is 'undirected'.
If the direction_option is 'bi_sep' or 'bi_fuse', n_etypes will be set to 1.

4.2.2 GGNNLayer Construction Function
------------------------------------
Similaer to ``GCNLayer``, ``GGNNLayer`` is a single-layer GGNN and its initial options are same as class ``GGNN``.
This module registers different GGNNLayerConv according to ``direction_option``.


4.2.3 GGNNLayerConv Construction Function
------------------------------------
We will take ``BiSepGGNNLayerConv`` as an example. The construction function performs the following steps:

1. Set options.
2. Register learnable parameters.
3. Reset parameters.

The aggregation and upate functions are formulated as:

.. math::
       h_{i}^{0} = [ x_i \| \mathbf{0} ]

       a_{i, \vdash}^{t} = \sum_{j\in\mathcal{N}_{\vdash }(i)} W_{\vdash} h_{j, \vdash}^{t}

       a_{i, \dashv}^{t} = \sum_{j\in\mathcal{N}_{\dashv }(i)} W_{\dashv} h_{j, \dashv}^{t}

       h_{i, \vdash}^{t+1} = \mathrm{GRU}_{\vdash}(a_{i, \vdash}^{t}, h_{i, \vdash}^{t})

       h_{i, \dashv}^{t+1} = \mathrm{GRU}_{\dashv}(a_{i, \dashv}^{t}, h_{i, \dashv}^{t})

As shown in the equations, node embeddings in both directions are conveyed separately.


.. code::

    class BiSepGGNNLayerConv(GNNLayerBase):
        def __init__(self, input_size, output_size, n_etypes=1, bias=True):
            super(BiSepGGNNLayerConv, self).__init__()
            self._input_size = input_size
            self._output_size = output_size
            self._n_etypes = n_etypes

            self.linears_in = nn.ModuleList(
                [nn.Linear(output_size, output_size) for _ in range(n_etypes)]
            )

            self.linears_out = nn.ModuleList(
                [nn.Linear(output_size, output_size) for _ in range(n_etypes)]
            )

            self.gru_in = nn.GRUCell(output_size, output_size, bias=bias)
            self.gru_out = nn.GRUCell(output_size, output_size, bias=bias)
            self.reset_parameters()

All learnable parameters and layers defined in this module are bidirectional, such as ``self.gru_in`` and ``self.gru_out``.


4.2.4 GGNN Forward Function
--------------------------
In NN module, ``forward()`` function does the actual message passing and computation. ``forward()`` takes a parameter ``GraphData`` as input.

The rest of the section takes a deep dive into the ``forward()`` function.

We first need to obatin the input graph node features and convert the ``GraphData`` to ``dgl.DGLGraph``. Then, we need to determine whether to expand ``feat`` according to ``self.use_edge_weight`` and whether to use edge weight according to ``self.direction_option``.

.. code::

    if self.n_etypes==1:
        graph.edge_features['etype'] = torch.tensor([0] * graph.get_edge_num(), dtype=torch.long, device=graph.device)

    node_feats = graph.node_features['node_feat']
    etypes = graph.edge_features['etype']
    if self.use_edge_weight:
        edge_weight = graph.edge_features['edge_weight']
        if self.direction_option == 'bi_fuse' or self.direction_option == 'bi_sep':
            reverse_edge_weight = graph.edge_features['reverse_edge_weight']
            edge_weight = (edge_weight, reverse_edge_weight)
        else:
            edge_weight = None

    dgl_graph = graph.to_dgl()

The following code actually performs message passing and feature updating.

.. code::

    if self.direction_option == 'undirected':
        node_embs = self.models(dgl_graph, node_feats, etypes, edge_weight)
    else:
        assert node_feats.shape[1] == self.input_size

        zero_pad = node_feats.new_zeros((node_feats.shape[0], self.output_size - node_feats.shape[1]))
        node_feats = torch.cat([node_feats, zero_pad], -1)

        feat_in = node_feats
        feat_out = node_feats

        for i in range(self.num_layers):
            feat_in = self.feat_drop(feat_in)
            feat_out = self.feat_drop(feat_out)
            h = self.models(dgl_graph, (feat_in, feat_out), etypes, edge_weight)
            feat_in = h[0]
            feat_out = h[1]

        if self.direction_option == 'bi_sep':
            node_embs = torch.cat([feat_in, feat_out], dim=-1)
        elif self.direction_option == 'bi_fuse':
            node_embs = feat_in
        else:
            raise RuntimeError('Unknown `bidirection` value: {}'.format(self.direction_option))

    graph.node_features['node_emb'] = node_embs
