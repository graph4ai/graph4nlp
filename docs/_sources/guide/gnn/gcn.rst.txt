.. _guide-gcn:

Graph Convolutional Networks
==============

Graph Convolutional Networks (`GCN <https://arxiv.org/abs/1609.02907>`_) is a typical example of spectral-based graph filters.
A multi-layer GCN is considered with the following layer-wise propagation rule using spectral graph theory:

.. math::
    \mathbf{H}^{(l)} = \sigma( {\tilde{D}}^{-\frac{1}{2}}{\tilde{A}}{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l-1)} \mathbf{W}^{(l-1)})

Here, :math:`\mathbf{W}^{(l-1)}` is a layer-specific trainable weight matrix and
:math:`\sigma(\cdot)` denotes an activation function.
:math:`\mathbf{H}^{(l)} \in \mathbb{R}^{n \times d}` is the activated node embeddings
at :math:`(l-1)`-th layer.


4.1.1 GCN Module Construction Function
---------------------------------------

The construction function performs the following steps:

1. Set options.
2. Register learnable parameters or submodules (``GCNLayer``).

.. code::

    class GCN(GNNBase):
	    def __init__(self,
	                 num_layers,
	                 input_size,
	                 hidden_size,
	                 output_size,
	                 direction_option='bi_sep',
	                 feat_drop=0.,
	                 gcn_norm='both',
	                 weight=True,
	                 bias=True,
	                 activation=None,
	                 allow_zero_in_degree=False,
	                 use_edge_weight=False,
	                 residual=True):
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
	            self.gcn_layers.append(GCNLayer(input_size,
	                                            hidden_size[0],
	                                            direction_option=self.direction_option,
	                                            feat_drop=feat_drop,
	                                            gcn_norm=gcn_norm,
	                                            weight=weight,
	                                            bias=bias,
	                                            activation=activation,
	                                            allow_zero_in_degree=allow_zero_in_degree,
	                                            residual=residual))

	        # hidden layers
	        for l in range(1, self.num_layers - 1):
	            # due to multi-head, the input_size = hidden_size * num_heads
	            self.gcn_layers.append(GCNLayer(hidden_size[l - 1],
	                                            hidden_size[l],
	                                            direction_option=self.direction_option,
	                                            feat_drop=feat_drop,
	                                            gcn_norm=gcn_norm,
	                                            weight=weight,
	                                            bias=bias,
	                                            activation=activation,
	                                            allow_zero_in_degree=allow_zero_in_degree,
	                                            residual=residual))
	        # output projection
	        self.gcn_layers.append(GCNLayer(hidden_size[-1] if self.num_layers > 1 else input_size,
	                                        output_size,
	                                        direction_option=self.direction_option,
	                                        feat_drop=feat_drop,
	                                        gcn_norm=gcn_norm,
	                                        weight=weight,
	                                        bias=bias,
	                                        activation=activation,
	                                        allow_zero_in_degree=allow_zero_in_degree,
	                                        residual=residual))

In construction function, one first needs to set the number of GCN layers and the data dimensions. For
general PyTorch module, the dimensions are usually input dimension,
output dimension and hidden dimension.

Besides data dimensions, a typical option for graph neural network is
direction option (``self.direction_option``). Direction option determines whether to use unidirectional (i.e., ``undirected``) or bidirectional (i.e., ``bi_sep`` and ``bi_fuse``) version of GCN.

``gcn_norm`` here is a callable function for feature normalization. In the
GCN paper, such normalization can be: ``right``, ``both``,
``none``.

``use_edge_weight`` represents whether to use edge weights when computing the node embeddings.

``residual`` represents whether to add residual connection between different GCN layers.


4.1.2 GCNLayer Construction Function
------------------------------------
``GCNLayer`` is a single-layer GCN and its initial options are same as class ``GCN``.
This module registers different GCNLayerConv according to ``direction_option``.

.. code::

    class GCNLayer(GNNLayerBase):
        def __init__(self,
                     input_size,
                     output_size,
                     direction_option='bi_sep',
                     feat_drop=0.,
                     gcn_norm='both',
                     weight=True,
                     bias=True,
                     activation=None,
                     allow_zero_in_degree=False,
                     residual=True):
            super(GCNLayer, self).__init__()
            if direction_option == 'undirected':
                self.model = UndirectedGCNLayerConv(input_size,
                                                    output_size,
                                                     feat_drop=feat_drop,
                                                     gcn_norm=gcn_norm,
                                                     weight=weight,
                                                     bias=bias,
                                                     activation=activation,
                                                     allow_zero_in_degree=allow_zero_in_degree,
                                                     residual=residual)
            elif direction_option == 'bi_sep':
                self.model = BiSepGCNLayerConv(input_size,
                                                 output_size,
                                                 feat_drop=feat_drop,
                                                 gcn_norm=gcn_norm,
                                                 weight=weight,
                                                 bias=bias,
                                                 activation=activation,
                                                 allow_zero_in_degree=allow_zero_in_degree,
                                                 residual=residual)
            elif direction_option == 'bi_fuse':
                self.model = BiFuseGCNLayerConv(input_size,
                                                 output_size,
                                                 feat_drop=feat_drop,
                                                 gcn_norm=gcn_norm,
                                                 weight=weight,
                                                 bias=bias,
                                                 activation=activation,
                                                 allow_zero_in_degree=allow_zero_in_degree,
                                                 residual=residual)
            else:
                raise RuntimeError('Unknown `direction_option` value: {}'.format(direction_option))


4.1.3 GCNLayerConv Construction Function
------------------------------------
We will take ``BiSepGCNLayerConv`` as an example. The construction function performs the following steps:

1. Set options.
2. Register learnable parameters.
3. Reset parameters.

The aggregation and upate functions are formulated as:

.. math::
        h_{i, \vdash}^{(l+1)} = \sigma(b^{(l)}_{\vdash} + \sum_{j\in\mathcal{N}_{\vdash}(i)}\frac{1}{c_{ij}}h_{j, \vdash}^{(l)}W^{(l)}_{\vdash})

        h_{i, \dashv}^{(l+1)} = \sigma(b^{(l)}_{\dashv} + \sum_{j\in\mathcal{N}_{\dashv}(i)}\frac{1}{c_{ij}}h_{j, \dashv}^{(l)}W^{(l)}_{\dashv})

As shown in the equations, node embeddings in both directions are conveyed separately.


.. code::

    class BiSepGCNLayerConv(GNNLayerBase):
        def __init__(self,
                     input_size,
                     output_size,
                     feat_drop=0.,
                     gcn_norm='both',
                     weight=True,
                     bias=True,
                     activation=None,
                     allow_zero_in_degree=False,
                     residual=True):
            super(BiSepGCNLayerConv, self).__init__()
            if gcn_norm not in ('none', 'both', 'right'):
                raise RuntimeError('Invalid gcn_norm value. Must be either "none", "both" or "right".'
                                   ' But got "{}".'.format(gcn_norm))
            self._input_size = input_size
            self._output_size = output_size
            self._gcn_norm = gcn_norm
            self._allow_zero_in_degree = allow_zero_in_degree
            self._feat_drop=nn.Dropout(feat_drop)

            if weight:
                self.weight_fw = nn.Parameter(torch.Tensor(input_size, output_size))
                self.weight_bw = nn.Parameter(torch.Tensor(input_size, output_size))
            else:
                self.register_parameter('weight_fw', None)
                self.register_parameter('weight_bw', None)

            if bias:
                self.bias_fw = nn.Parameter(torch.Tensor(output_size))
                self.bias_bw = nn.Parameter(torch.Tensor(output_size))
            else:
                self.register_parameter('bias_fw', None)
                self.register_parameter('bias_bw', None)

            if residual:
                if self._input_size != output_size:
                    self.res_fc_fw = nn.Linear(
                        self._input_size, output_size, bias=True)
                    self.res_fc_bw = nn.Linear(
                        self._input_size, output_size, bias=True)
                else:
                    self.res_fc_fw = self.res_fc_bw = nn.Identity()
            else:
                self.register_buffer('res_fc_fw', None)
                self.register_buffer('res_fc_bw', None)

            self.reset_parameters()

            self._activation = activation

All learnable parameters and layers defined in this module are bidirectional, such as ``self.weight_fw`` and ``self.weight_bw``.

Similarly, the aggregation and upate functions of ``BiFuseGCNLayerConv`` are formulated as:

.. math::
        h_{i, \vdash}^{(l+1)} = \sigma(b^{(l)}_{\vdash} + \sum_{j\in\mathcal{N}_{\vdash}(i)}\frac{1}{c_{ij}}h_{j}^{(l)}W^{(l)}_{\vdash})

        h_{i, \dashv}^{(l+1)} = \sigma(b^{(l)}_{\dashv} + \sum_{j\in\mathcal{N}_{\dashv}(i)}\frac{1}{c_{ij}}h_{j}^{(l)}W^{(l)}_{\dashv})

        r_{i}^{l} = \sigma (W_{f}[h_{i, \vdash}^{l};h_{i, \dashv}^{l};
                h_{i, \vdash}^{l}*h_{i, \dashv}^{l};
                h_{i, \vdash}^{l}-h_{i, \dashv}^{l}])

Node embeddings in both directions are fused in every layer. The construction code of ``BiFuseGCNLayerConv`` is as follows:

.. code::

    class BiFuseGCNLayerConv(GNNLayerBase):

        def __init__(self,
                     input_size,
                     output_size,
                     feat_drop=0.,
                     gcn_norm='both',
                     weight=True,
                     bias=True,
                     activation=None,
                     allow_zero_in_degree=False,
                     residual=True):
            super(BiFuseGCNLayerConv, self).__init__()
            if gcn_norm not in ('none', 'both', 'right'):
                raise RuntimeError('Invalid gcn_norm value. Must be either "none", "both" or "right".'
                                   ' But got "{}".'.format(gcn_norm))
            self._input_size = input_size
            self._output_size = output_size
            self._gcn_norm = gcn_norm
            self._allow_zero_in_degree = allow_zero_in_degree
            self._feat_drop=nn.Dropout(feat_drop)

            if weight:
                self.weight_fw = nn.Parameter(torch.Tensor(input_size, output_size))
                self.weight_bw = nn.Parameter(torch.Tensor(input_size, output_size))
            else:
                self.register_parameter('weight_fw', None)
                self.register_parameter('weight_bw', None)

            if bias:
                self.bias_fw = nn.Parameter(torch.Tensor(output_size))
                self.bias_bw = nn.Parameter(torch.Tensor(output_size))
            else:
                self.register_parameter('bias_fw', None)
                self.register_parameter('bias_bw', None)

            self.reset_parameters()

            self._activation = activation

            self.fuse_linear = nn.Linear(4 * output_size, output_size, bias=True)

            if residual:
                if self._input_size != output_size:
                    self.res_fc = nn.Linear(
                        self._input_size, output_size, bias=True)
                else:
                    self.res_fc = nn.Identity()
            else:
                self.register_buffer('res_fc', None)

4.1.4 GCN Forward Function
--------------------------
In NN module, ``forward()`` function does the actual message passing and computation. ``forward()`` takes a parameter ``GraphData`` as input.

The rest of the section takes a deep dive into the ``forward()`` function.

We first need to obatin the input graph node features and convert the ``GraphData`` to ``dgl.DGLGraph``. Then, we need to determine whether to expand ``feat`` according to ``self.use_edge_weight`` and whether to use edge weight according to ``self.direction_option``.

.. code::

    feat = graph.node_features['node_feat']
    dgl_graph = graph.to_dgl()

    if self.direction_option == 'bi_sep':
        h = [feat, feat]
    else:
        h = feat

    if self.use_edge_weight:
        edge_weight = graph.edge_features['edge_weight']
        if self.direction_option != 'undirected':
            reverse_edge_weight = graph.edge_features['reverse_edge_weight']
        else:
            reverse_edge_weight = None
    else:
        edge_weight = None
        reverse_edge_weight = None

The following code actually performs message passing and feature updating.

.. code::

    for l in range(self.num_layers - 1):
        h = self.gcn_layers[l](dgl_graph, h, edge_weight=edge_weight, reverse_edge_weight=reverse_edge_weight)
        if self.direction_option == 'bi_sep':
            h = [each.flatten(1) for each in h]
        else:
            h = h.flatten(1)

    logits = self.gcn_layers[-1](dgl_graph, h)

    if self.direction_option == 'bi_sep':
        logits = torch.cat(logits, -1)
    else:
        pass

    graph.node_features['node_emb'] = logits
