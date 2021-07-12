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
Below is an example to call the API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN

    clf = FeedForwardNN(32, # input size
                        2, # output size
                        [32], # list of hidden size for each FFN layer
                        graph_pool_type='avg_pool')




.. _guide-AvgPooling

AvgPooling
-----------------

This is the average pooling module which applies average pooling over the nodes in the graph.
Below is an example to call the API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.graph_classification import AvgPooling
    graph_pool = AvgPooling()
    graph_emb = graph_pool(graph_data, 'node_emb') # Input: the graph data, the feature field name, output: the graph embedding.




.. _guide-MaxPooling

MaxPooling
-----------------

This is the max pooling module which applies max pooling over the nodes in the graph.
Below is an example to call the API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.graph_classification import MaxPooling
    graph_pool = MaxPooling()
    graph_emb = graph_pool(graph_data, 'node_emb') # Input: the graph data, the feature field name, output: the graph embedding.

