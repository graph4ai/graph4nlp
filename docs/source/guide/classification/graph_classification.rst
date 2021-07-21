.. _guide-graph_classification:

Graph Classification
===========

Graph classification is a downstream classification task conducted at the graph level. Once node representations are learned by a GNN,
one can obtain the graph-level representation and then perform graph-level classification. To facilitate the graph classification task,
we provide commonly used implementations of the graph classification prediction modules.



.. _guide-FeedForwardNN

FeedForwardNN
-----------------

This is a high-level graph classification prediction module which consists of a graph pooling component and a multilayer perceptron (MLP).
Users can specify important hyperparameters such as ``input_size``, ``num_class`` and ``hidden_size`` (i.e., list of hidden sizes for each dense layer).
The ``FeedForwardNN`` class calls the ``FeedForwardNNLayer`` API which implments MLP.

.. code-block:: python

    class FeedForwardNN(GraphClassifierBase):
        def __init__(self, input_size, num_class, hidden_size, activation=None, graph_pool_type='max_pool', **kwargs):
            super(FeedForwardNN, self).__init__()

            if not activation:
                activation = nn.ReLU()

            if graph_pool_type == 'avg_pool':
                self.graph_pool = AvgPooling()
            elif graph_pool_type == 'max_pool':
                self.graph_pool = MaxPooling(**kwargs)
            else:
                raise RuntimeError('Unknown graph pooling type: {}'.format(graph_pool_type))

            self.classifier = FeedForwardNNLayer(input_size, num_class, hidden_size, activation)




.. _guide-AvgPooling

AvgPooling
-----------------

This is the average pooling module which applies average pooling over the nodes in the graph.
It takes batched ``GraphData`` as input and returns a feature tensor containing a vector for each graph in the batch.

.. code-block:: python

    class AvgPooling(PoolingBase):
        def __init__(self):
            super(AvgPooling, self).__init__()

        def forward(self, graph, feat):
            graph_list = from_batch(graph)
            output_feat = []
            for g in graph_list:
                output_feat.append(g.node_features[feat].mean(dim=0))

            output_feat = torch.stack(output_feat, 0)

            return output_feat




.. _guide-MaxPooling

MaxPooling
-----------------

This is the max pooling module which applies max pooling over the nodes in the graph.
It takes batched ``GraphData`` as input and returns a feature tensor containing a vector for each graph in the batch.
An optional linear projection can be applied to node embeddings before conducting max pooling.

.. code-block:: python

    class MaxPooling(PoolingBase):
        def __init__(self, dim=None, use_linear_proj=False):
            super(MaxPooling, self).__init__()
            if use_linear_proj:
                assert dim is not None, "dim should be specified when use_linear_proj is set to True"
                self.linear = nn.Linear(dim, dim, bias=False)
            else:
                self.linear = None

        def forward(self, graph, feat):
            graph_list = from_batch(graph)
            output_feat = []
            for g in graph_list:
                feat_tensor = g.node_features[feat]
                if self.linear is not None:
                    feat_tensor = self.linear(feat_tensor)

                output_feat.append(torch.max(feat_tensor, dim=0)[0])

            output_feat = torch.stack(output_feat, 0)

            return output_feat
