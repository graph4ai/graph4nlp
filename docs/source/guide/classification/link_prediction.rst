.. _guide-link_prediction:

Link_prediction:
===================

Link prediction is a downstream task that are normally observed in the GNN-based NLP tasks, such as relation extract and amr parsing. The process is about classify the label of each edge or predict whether there is an edge between a pair of nodes based on the node embeddings that learnt from the GNNs modules.
To facilitate the implementation of link prediction task, we provide both high-level and low-level APIs to users to easily define a multi-layer link prediction function. Besides, for each level's APIs, we support three popularly used node classifiers, namely, ConcatFeedForwardNN, ElementSum, and StackedElementProd.



.. _guide-ConcatFeedForwardNN

ConcatFeedForwardNN:
==================

This function is based on the feedforward layer. To predict the edge between a pair of nodes, the embeddings of the two nodes is first concatenated and then inputed into the Feedforward neural network. The low-level function defines a two-layer classifier with the input of node embedding tensor and the output of legit tensor after classification. The user can specify the index of edge (represented as tuple of nodes pair indexes) that needs prediction. If not specified, the classifier will be applied to all pair of nodes. Below is an example to call the ConcatFeedForwardNNLayer API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.link_prediction import ConcatFeedForwardNNLayer
    From torch import nn

    layer = ConcatFeedForwardNNLayer(32, # input size of node embedding
            16, # hidden size of linear layer
            2, # The number of node category for classification
            nn.ReLU())

After successfully define the ConcatFeedForwardNNLayer, we can apply this function to get the prediction results (logins tensor) of specific pair of nodes based on the node embedding, as shown in the below example.

.. code-block:: python

    logits = layer(node_emb, #node embedding in format of tensor
             edge_idx, # a list of index of nodes that needs classification;Example: [(1,2),(1,0),(2,9)])

   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for link prediction, where the input and output are both the graph in type of `GraphData`. The node embedding tensor should be stored in the node feature field named "node_emb"  in the input_graph for prediction. The computed logit tensor for each pair of nodes in the graph are stored in the edge feature field named "edge_logits". The input graph can be either batched graph or original single graph. Below is an example to define and call the ConcatFeedForwardNN API.


.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.link_prediction import ConcatFeedForwardNN

    classifier = ConcatFeedForwardNN(32, # input size of node embeding
            16, # hidden size of linear layer
            2, # The number of node category for classification
            nn.ReLU())

    output_graph = classifier (input_graph)




.. _guide-ElementSum

ElementSum:
==================

This function is also based on the feedforward layer. To predict the edge between a pair of nodes, the embeddings of the two nodes is first inputted into feedforward neural network and get the updated embedding. Then the element sum operation is conducted on the two embedding for the final prediction. The low-level function defines a two-layer classifier with the input of node embedding tensor and the output of legit tensor after prediction. The user can specify the index of edge (represented as tuple of nodes pair indexes) that needs prediction. If not specified, the classifier will be applied to all pair of nodes. Below is an example to call the ElementSumLayer API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.link_prediction import ElementSumLayer
    From torch import nn

    layer = ElementSumLayer(32, # input size of node embedding
            16, # hidden size of linear layer
            2, # The number of node category for classification
            nn.ReLU())

After successfully define the ElementSumLayer, we can apply this function to get the prediction results (logins tensor) of specific pair of nodes based on the node embedding, as shown in the below example.

.. code-block:: python

    logits = layer(node_emb, #node embedding in format of tensor
             edge_idx, # a list of index of nodes that needs classification;Example: [(1,2),(1,0),(2,9)])

   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for link prediction, where the input and output are both the graph in type of `GraphData`. The node embedding tensor should be stored in the node feature field named "node_emb"  in the input_graph for prediction. The computed logit tensor for each pair of nodes in the graph are stored in the edge feature field named "edge_logits". The input graph can be either batched graph or original single graph. Below is an example to define and call the ElementSum API.


.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.link_prediction import ElementSum

    classifier = ElementSum(32, # input size of node embedding
            16, # hidden size of linear layer
            2, # The number of node category for classification
            nn.ReLU())

    output_graph = classifier (input_graph)





.. _guide-StackedElementProd

StackedElementProd:
==================

This function is also based on the feedforward layer and designed for a multi-layer GNN encoder. To predict the edge between a pair of nodes, the products of the embeddings of two nodes at each GNN-layer will be concatenated. Then the concatenation will be finally inputted into the feedforward neural network for the final prediction. The low-level function defines a classifier layer with the input of node embedding list (each element in the list refers to a node embedding tensor at each layer) and the output of legit tensor after prediction. The user can specify the index of edge (represented as tuple of nodes pair indexes) that needs prediction. If not specified, the classifier will be applied to all pair of nodes. Below is an example to call the StackedElementProdLayer API.

.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.link_prediction import StackedElementProdLayer
    From torch import nn

    layer = StackedElementProdLayer(32, # input size of node embedding
            16, # hidden size of linear layer
            2, # The number of node category for classification
            2, #num of channels for node embedding
            nn.ReLU())

After successfully define the StackedElementProdLayer, we can apply this function to get the prediction results (logins tensor) of specific pair of nodes based on the node embedding, as shown in the below example.

.. code-block:: python

    logits = layer(node_emb_list, #node embedding in format of tensor
             edge_idx, # a list of index of nodes that needs classification;Example: [(1,2),(1,0),(2,9)])

   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for link prediction, where the input and output are both the graph in type of `GraphData`. The node embedding tensor at channel `N` should be stored in the node feature field named "node_emb_<N>"  in the input_graph for prediction. The computed logit tensor for each pair of nodes in the graph are stored in the edge feature field named "edge_logits". The input graph can be either batched graph or original single graph. Below is an example to define and call the StackedElementProd API.


.. code-block:: python

    from graph4nlp.pytorch.modules.prediction.classification.link_prediction import StackedElementProd

    classifier = StackedElementProd(32, # input size of node embedding
            16, # hidden size of linear layer
            2, # The number of node category for classification
            2, # The number of channels for node embedding
            nn.ReLU())

    output_graph = classifier (input_graph)
