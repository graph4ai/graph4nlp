.. _guide-graphsage:

GraphSAGE
===========


GraphSAGE (`GraphSAGE <https://arxiv.org/pdf/1706.02216.pdf>`__) is a framework for inductive representation learning on large graphs. GraphSAGE is used to generate low-dimensional vector representations for nodes, and is especially useful for graphs that have rich node attribute information. The math operation of GraphSAGE is represented as below:

.. math::
        h_{\mathcal{N}(i)}^{(l+1)}  = \mathrm{aggregate}\left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
        
        h_{i}^{(l+1)}  = \sigma \left(W \cdot \mathrm{concat}(h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)
        
        h_{i}^{(l+1)}  = \mathrm{norm}(h_{i}^{l})

We provide high level APIs to users to easily define a multi-layer GraphSage model. Besides, we also support both regular GraphSAGE and bidirectional versions including `GraphSAGE-BiSep <https://arxiv.org/abs/1808.07624>`__
and `GraphSAGE BiFuse <https://arxiv.org/abs/1908.04942>`__.


GraphSAGE Module Construction Function
--------------------------------------

The construction function performs the following steps:

1. Set options.
2. Register learnable parameters or submodules (``GraphSAGELayer``).

.. code::

     class GraphSAGE(GNNBase):
         def __init__(self, num_layers, input_size, hidden_size, output_size, aggregator_type, direction_option='undirected', feat_drop=0., bias=True, norm=None, activation=None, use_edge_weight=False):
        super(GraphSAGE, self).__init__()
        self.use_edge_weight=use_edge_weight
        self.num_layers = num_layers
        self.direction_option = direction_option
        self.GraphSAGE_layers = nn.ModuleList()


Users can select the number of layers in the GraphSAGE module. If ``num_layers`` is larger  Than 1, then ``hidden_size`` should be a list of int values. Based on the values of ``num_layers``, we construct the module as

..code::

        if self.num_layers>1:
            # input projection
            self.GraphSAGE_layers.append(GraphSAGELayer(input_size,
                                            hidden_size[0],
                                            aggregator_type,
                                            direction_option=self.direction_option,
                                            feat_drop=feat_drop,
                                            bias=bias,
                                            norm=norm,
                                            activation=activation))


If ``num_layers`` is larger than 1, while the ``hidden_size`` is an int format value, we assume that all the hidden layers have the same size as

..code::

        for l in range(1, self.num_layers - 1):
            # due to multi-head, the input_size = hidden_size * num_heads
            self.GraphSAGE_layers.append(GraphSAGELayer(hidden_size[l - 1],
                                            hidden_size[l],
                                            aggregator_type,
                                            direction_option=self.direction_option,
                                            feat_drop=feat_drop,
                                            bias=bias,
                                            norm=norm,
                                            activation=activation))


GraphSAGE Module Forward Function
--------------------------------------
In NN module, ``forward()`` function does the actual message passing and computation. ``forward()`` takes a parameter ``GraphData`` as input.

The rest of the section takes a deep dive into the ``forward()`` function.

We first need to obatin the input graph node features and convert the ``GraphData`` to ``dgl.DGLGraph``. Then, we need to determine whether to expand ``feat`` according to ``self.use_edge_weight`` and whether to use edge weight according to ``self.direction_option``. 


.. code::

        h=graph.node_features['node_feat'] #get the node feature tensor from graph
        g = graph.to_dgl() #transfer the current NLPgraph to DGL graph
        edge_weight=None
        reverse_edge_weight=None

        if self.use_edge_weight==True:
            edge_weight = graph.edge_features['edge_weight']
            reverse_edge_weight = graph.edge_features['reverse_edge_weight']

Then we call the low-level GraphSAGE layer to complete the message passing operation. The updated node embedding will be stored back into the node field ``node_emb`` of GraphData and the final output is the GraphData.

.. code::

        if self.num_layers>1:
          for l in range(0,self.num_layers - 1):
              h = self.GraphSAGE_layers[l](g, h, edge_weight,reverse_edge_weight)

        logits = self.GraphSAGE_layers[-1](g, h, edge_weight,reverse_edge_weight)

        if self.direction_option == 'bi_sep':
            logits = torch.cat(logits, -1)
        else:
            logits = logits

        graph.node_features['node_emb']=logits #put the results into the NLPGraph





GraphSAGELayer Construction Function
------------------------------------

To make the utilization of GraphSAGE more felxbible, we also provide the low-level implementation of GraphSAGE layer. Below is how to define the ``GraphSAGELayer`` API.

.. code::

    class GraphSAGELayer(GNNLayerBase):
        def __init__(self, input_size, output_size, aggregator_type, direction_option='undirected', feat_drop=0., bias=True, norm=None, activation=None):
        super(GraphSAGELayer, self).__init__()

Consider we have three options for direction of embeddings, next step is to select the direction type based on ``direct_option``. We take the ``undirected`` as an example.

.. code::

        if direction_option == 'undirected':
            self.model = UndirectedGraphSAGELayerConv(input_size,
                                        output_size,
                                        aggregator_type,
                                        feat_drop=feat_drop,
                                        bias=bias,
                                        norm=norm,
                                        activation=activation)


GraphSAGELayer Forward Function
------------------------------------

After define a GraphSAGE layer, we can use it to get the node embedding for the input graph. The generated embedding is the output of this layer, as shown in the below example:

.. code::

    def forward(self, graph, feat, edge_weight=None,reverse_edge_weight=None):
        return self.model(graph, feat, edge_weight,reverse_edge_weight)




GraphSAGELayerConv Construction Function
------------------------------------

Then let us dive deep to see how the message passing of ``GraphSAGELayerConv`` for different direction options are implemented.  As an example, we introduce the details of the ``UndirectedGraphSAGELayerConv``. The construction function performs the following steps:

1. Set options.
2. Register learnable parameters.
3. Reset parameters.

.. code::

    def __init__(self, in_feats, out_feats, aggregator_type, feat_drop=0., bias=True, norm=None, activation=None):
        super(UndirectedGraphSAGELayerConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

There are three aggregation types for aggregating the messages passing to each node, namely,  ``mean``, ``pool``, ``lstm``, and ``gcn``. And the end of the above code, the parameters are reset.



GraphSAGELayerConv Forwards Function
------------------------------------

The message passing operation have four options considering four aggregation types. Here we take the ``list`` type as an example.

.. code::

        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src

            if edge_weight is None:
                graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            else:
               graph.edata['edge_weight']=edge_weight
               graph.update_all(fn.u_mul_e('h', 'edge_weight','m'), self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']

We could find that the above implementation also consider the situation of using the ``edge_weight``.

After the message passing and aggregation of the messages, we finally update the embedding of nodes and make them as the final outputs as

.. code::

        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)


