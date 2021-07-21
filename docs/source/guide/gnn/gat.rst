.. _guide-gat:

Graph Attention Networks
===========


The Graph Attention Network (`GAT <https://arxiv.org/abs/1710.10903>`__) aims to learn edge weights for the input binary adjacency matrix by introducing
the multi-head attention mechanism to the GNN architecture when performing message passing.
We provide high level APIs to users to easily define a multi-layer GAT model. Besides, we support both
regular GAT and bidirectional versions including `GAT-BiSep <https://arxiv.org/abs/1808.07624>`__
and `GAT-BiFuse <https://arxiv.org/abs/1908.04942>`__. The math operation of GAT is represented as below:

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`.


Where the attention matrix is computed as below:

    .. math::
        \alpha_{ij}^{l} = \mathrm{softmax_i} (e_{ij}^{l})\\
        e_{ij}^{l} = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)


GAT Module Construction Function
--------------------------------------

The construction function performs the following steps:

1. Set options.
2. Register learnable parameters or submodules (``GATLayer``).

.. code-block:: python

    class GAT(GNNBase):
        def __init__(self, num_layers, input_size, hidden_size, output_size, heads,
            direction_option='bi_sep', feat_drop=0., attn_drop=0., negative_slope=0.2,
            residual=False, activation=None, allow_zero_in_degree=False):
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
                self.gat_layers.append(GATLayer(input_size, hidden_size[0], heads[0],
                                        direction_option=self.direction_option,
                                        feat_drop=feat_drop, attn_drop=attn_drop,
                                        negative_slope=negative_slope, residual=residual,
                                        activation=activation, allow_zero_in_degree=allow_zero_in_degree))

            # hidden layers
            for l in range(1, self.num_layers - 1):
                # due to multi-head, the input_size = hidden_size * num_heads
                self.gat_layers.append(GATLayer(hidden_size[l - 1] * heads[l - 1], hidden_size[l],
                                            heads[l], direction_option=self.direction_option,
                                            feat_drop=feat_drop, attn_drop=attn_drop,
                                            negative_slope=negative_slope, residual=residual,
                                            activation=activation, allow_zero_in_degree=allow_zero_in_degree))
            # output projection
            self.gat_layers.append(GATLayer(hidden_size[-1] * heads[-2] if self.num_layers > 1 else input_size,
                                        output_size, heads[-1], direction_option=self.direction_option,
                                        feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                                        residual=residual, activation=None, allow_zero_in_degree=allow_zero_in_degree))


In construction function, one first needs to set the number of GAT layers and the data dimensions (i.e., ``input_size``, ``hidden_size``, ``output_size``).
For GAT, ``heads`` is also an important parameter. One should also specify ``direction_option`` which determines whether to use
unidirectional (i.e., undirected) or bidirectional (i.e., `bi_sep` and `bi_fuse`) version of GAT.



GAT Module Forward Function
--------------------------------------
In NN module, ``forward()`` function does the actual message passing and computation. ``forward()`` takes a parameter ``GraphData`` as input.
The updated node embedding will be stored back into the node field ``node_emb`` of GraphData and the final output is the GraphData.


.. code-block:: python

    def forward(self, graph):
        feat = graph.node_features['node_feat']
        dgl_graph = graph.to_dgl()

        if self.direction_option == 'bi_sep':
            h = [feat, feat]
        else:
            h = feat

        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](dgl_graph, h)
            if self.direction_option == 'bi_sep':
                h = [each.flatten(1) for each in h]
            else:
                h = h.flatten(1)

        # output projection
        logits = self.gat_layers[-1](dgl_graph, h)

        if self.direction_option == 'bi_sep':
            logits = [each.mean(1) for each in logits]
            logits = torch.cat(logits, -1)
        else:
            logits = logits.mean(1)

        graph.node_features['node_emb'] = logits

        return graph



GATLayer Construction Function
------------------------------------

To make the utilization of GAT more felxbible, we also provide the low-level implementation of GAT layer. Similarly to high-level API, users can specify ``direction_option`` which determines whether to use
unidirectional (i.e., undirected) or bidirectional (i.e., `bi_sep` and `bi_fuse`) GAT.

.. code-block:: python

    class GATLayer(GNNLayerBase):
        def __init__(self, input_size, output_size, num_heads, direction_option='bi_sep', feat_drop=0.,
            attn_drop=0., negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False):
        super(GATLayer, self).__init__()
        if num_heads >  1 and residual:
            residual = False
            import warnings
            warnings.warn("The residual option must be False when num_heads > 1")
        if direction_option == 'undirected':
            self.model = UndirectedGATLayerConv(input_size, output_size, num_heads, feat_drop=feat_drop,
                                attn_drop=attn_drop, negative_slope=negative_slope, residual=residual,
                                activation=activation, allow_zero_in_degree=allow_zero_in_degree)
        elif direction_option == 'bi_sep':
            self.model = BiSepGATLayerConv(input_size, output_size, num_heads, feat_drop=feat_drop,
                                attn_drop=attn_drop, negative_slope=negative_slope, residual=residual,
                                activation=activation, allow_zero_in_degree=allow_zero_in_degree)
        elif direction_option == 'bi_fuse':
            self.model = BiFuseGATLayerConv(input_size, output_size, num_heads, feat_drop=feat_drop,
                                attn_drop=attn_drop, negative_slope=negative_slope, residual=residual,
                                activation=activation, allow_zero_in_degree=allow_zero_in_degree)
        else:
            raise RuntimeError('Unknown `direction_option` value: {}'.format(direction_option))
