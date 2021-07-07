.. _guide-node_classification:

Node_classification:
===================

Node classification is a downstream task that are normally observed in the GNN-based NLP tasks, such as sequence labeling and name entity recognition. The process is about classify the label of the each node in the graph based on the node embeddings that learnt from the GNNs modules.
To facilitate the implementation of node classification task, we provide both high-level and low-level APIs to users to easily define a multi-layer node classification function. Besides, for each level's APIs, we support two popularly used node classifiers, one is based on the BiLSTM and feedforward neural network (BiLSTMFeedForwardNN), another one is based on simple feedforward neural network (FeedForwardNN).



.. _guide-BiLSTMFeedForwardNN

BiLSTMFeedForwardNN:
==================

This function is based on a combination of the BiLSTM layer and a feedforward layer. The low-level function defines the a single layer classifier with the input of node embedding tensor and the output of legit tensor after classification. The user can specify the index of nodes that needs to be classified. If not specified, the classifier will be applied to all the nodes. Below is an example to call the BiLSTMFeedForwardNNLayer API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification import BiLSTMFeedForwardNNLayer

    layer = BiLSTMFeedForwardNNLayer(32, # input size of node embeding
            16, # hidden size of linear layer
            2, # The number of node catrgoriey for classification
            th.device ('cuda'), #whether use 'cpu' or 'gpu'
            Dropout = 0, # the dropout rate of bilstm layer)

After successfully define the BiLSTMFeedForwardNNLayer, we can apply this function to get the classification results (logins tensor) of specific nodes based on the node embedding, as shown in the below example.

.. code-block:: python

    logits = layer(node_emb, #node embedding in format of tensor
             node_idx, # a list of index of nodes that needs classification; Example: [1,3,5])

   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for node classification, where the input and output are both the graph in type of `GraphData`. The node embedding tensor should be stored in the node feature field named "node_emb"  in the input_graph for classification. The computed logit tensor for each nodes in the graph are stored in the node feature field named "node_logits". The input graph can be either batched graph or original single graph. Below is an example to define and call the BiLSTMFeedForwardNN API.


.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification import BiLSTMFeedForwardNN

    classifier = BiLSTMFeedForwardNN(32, # input size of node embeding
            16, # hidden size of linear layer
            2, # The number of node catrgoriey for classification
            Dropout = 0, # the dropout rate of bilstm layer)

    output_graph = classifier (input_graph)




.. _guide-FeedForwardNN

FeedForwardNN:
==================

This function is based on a combination of several feedforward layer. The low-level function defines the classifier layers with the input of node embedding tensor and the output of legit tensor after classification. The user can specify the index of nodes that needs to be classified. If not specified, the classifier will be applied to all the nodes. Below is an example to call the single FeedForwardNNLayer API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification import FeedForwardNNLayer
    From torch import nn

    layer = FeedForwardNNLayer(32, # input size of node embedding
            16, # hidden size of linear layer, Example for two layers's FeedforwardNN: [50, 20]           
            2, # The number of node categories for classification
            nn.ReLU())

After successfully define the ForwardNNLayer, we can apply this function to get the classification results (logins tensor) of specific nodes based on the node embedding, as shown in the below example.

.. code-block:: python

    logits = layer(node_emb, #node embedding in format of tensor
             node_idx, # a list of index of nodes that needs classification; Example: [1,3,5])

   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for multi-layer node classification, where the input and output are both the graph in type of `GraphData`. The node embedding tensor should be stored in the node feature field named "node_emb"  in the input_graph for classification. The computed logit tensor for each nodes in the graph are stored in the node feature field named "node_logits". The input graph can be either batched graph or original single graph. Below is an example to define and call the FeedForwardNN API.


.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification import FeedForwardNN

    classifier = FeedForwardNN(32, # input size of node embeding
            16, # hidden size of linear layer
            2, # The number of node catrgoriey for classification
            nn.ReLU() # the dropout rate of bilstm layer)

    output_graph = classifier (input_graph)
