.. _guide-link_prediction:

Link Prediction
===================

Link prediction is a downstream task that are normally observed in the GNN-based NLP tasks, such as relation extract and amr parsing. The process is about classify the label of each edge or predict whether there is an edge between a pair of nodes based on the node embeddings that learnt from the GNNs modules.
To facilitate the implementation of link prediction task, we provide both high-level and low-level APIs to users to easily define a multi-layer link prediction function. Besides, for each level's APIs, we support three popularly used node classifiers, namely, ConcatFeedForwardNN, ElementSum, and StackedElementProd.




ConcatFeedForwardNN
--------------------

This function is based on the feedforward layer. To predict the edge between a pair of nodes, the embeddings of the two nodes is first concatenated and then inputed into the Feedforward neural network. The low-level function defines a two-layer classifier with the input of node embedding tensor and the output of legit tensor after classification. The user can specify the index of edge (represented as tuple of nodes pair indexes) that needs prediction. If not specified, the classifier will be applied to all pair of nodes. Below is how the ConcatFeedForwardNNLayer module.



.. code-block:: python

   class ConcatFeedForwardNNLayer(LinkPredictionLayerBase):   
      def __init__(self, input_size, hidden_size, num_class, activation=nn.ReLU()):        
        super(ConcatFeedForwardNNLayer, self).__init__() 
            
        #build the linear module list
        self.activation=activation
        self.ffnn_all1 = nn.Linear(2*input_size, hidden_size)
        self.ffnn_all2 = nn.Linear(hidden_size, num_class)

As shown above, there are two feed forward layers implemented. The ``num_class`` is the number of edge types. If there is no edge type, just make ``num_class`` as 2. In this way, we could predict whether there is an edge between any pair of nodes.

After successfully define the ConcatFeedForwardNNLayer, we implementt the ``forward()`` part to get the prediction results (logins tensor) of specific pair of nodes based on the node embedding, as shown in the below example. ``edge_idx`` is a list and each element is a tuple of two node index. 

.. code::

          src_idx=torch.tensor([tuple_idx[0] for tuple_idx in edge_idx])    
          dst_idx=torch.tensor([tuple_idx[1] for tuple_idx in edge_idx]) 

If ``edge_idx`` is not given, the module will return the prediction logits of all pair of nodes.

.. code::
  
          num_node=node_emb.shape[0] #get the index list for all the node pairs
          node_idx_list=[idx for idx in range(num_node)]
          src_idx=torch.tensor(node_idx_list).view(-1,1).repeat(1,num_node).view(-1)
          dst_idx=torch.tensor(node_idx_list).view(1,-1).repeat(num_node,1).view(-1)

Then the final prediction is conducted based on ``src_emb`` and ``dst_emb`` as

.. code::

        src_emb = node_emb[src_idx, :] # input the source node embeddings into ffnn
        dst_emb = node_emb[dst_idx, :]  # input the destinate node embeddings into ffnn
        fused_emb=self.ffnn_all1(torch.cat([src_emb, dst_emb], dim=1))


   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for link prediction, where the input and output are both the graph in type of `GraphData`. The node embedding tensor should be stored in the node feature field named "node_emb"  in the input_graph for prediction. The computed logit tensor for each pair of nodes in the graph are stored in the edge feature field named "edge_logits". The ``forward`` part of ``ConcatFeedForwardNN`` module is implement as

.. code::
        num_node=node_emb.shape[0]
        node_idx_list=[idx for idx in range(num_node)]
        src_idx=torch.tensor(node_idx_list).view(-1,1).repeat(1,num_node).view(-1)
        dst_idx=torch.tensor(node_idx_list).view(1,-1).repeat(num_node,1).view(-1)
        
        input_graph.add_edges(src_idx,dst_idx)
        input_graph.edge_features['logits']=self.classifier(node_emb)

Here the ``self.classifier`` is defined by ``ConcatFeedForwardNNLayer`` mentioned above.



ElementSum
------------

This function is also based on the feedforward layer. To predict the edge between a pair of nodes, the embeddings of the two nodes is first inputted into feedforward neural network and get the updated embedding. Then the element sum operation is conducted on the two embedding for the final prediction. The low-level function defines a two-layer classifier with the input of node embedding tensor and the output of legit tensor after prediction. The user can specify the index of edge (represented as tuple of nodes pair indexes) that needs prediction. If not specified, the classifier will be applied to all pair of nodes. 

Below is how the ElementSumLayer module constructed.

.. code::
	class ElementSumLayer(LinkPredictionLayerBase):
  	    def __init__(self, input_size, hidden_size,num_class, activation=nn.ReLU()):        
    	    super(ElementSumLayer, self).__init__() 
    	        
     	    #build the linear module list
     	    self.activation=activation
     	    self.ffnn_src = nn.Linear(input_size, hidden_size)
     	    self.ffnn_dst = nn.Linear(input_size, hidden_size)
      	    self.ffnn_all = nn.Linear(hidden_size, num_class)

As shown above. Three linear layers are defined. Two are for embeddings from source nodes and destinated node, the other one is for the final aggregation step.  


After successfully define the module, we implement the ``forward()`` part to get the prediction results (logins tensor) of specific pair of nodes based on the node embedding, as shown in the below example. Similar to the ``ConcatFeedForwardNNLayer``, we first get the ``src_idx`` and ``dst_idx``. Based on them, the final prediction is conducted as

.. code::
        scr_emb = self.ffnn_src(node_emb[src_idx, :])  # input the source node embeddings into ffnn
        dst_emb = self.ffnn_dst(node_emb[dst_idx, :])  # input the destinate node embeddings into ffnn

Then the final output is

.. code::
  
        self.ffnn_all(self.activation(scr_emb+dst_emb))


   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function here, where the input and output are both the graph in type of `GraphData`. The node embedding tensor should be stored in the node feature field named "node_emb"  in the input_graph for prediction. The computed logit tensor for each pair of nodes in the graph are stored in the edge feature field named "edge_logits". The ``forward`` part of ``ElementSum`` is the same to that of ``ConcatFeedForwardNN``.








StackedElementProd
------------------


This function is also based on the feedforward layer and designed for a multi-layer GNN encoder. To predict the edge between a pair of nodes, the products of the embeddings of two nodes at each GNN-layer will be concatenated. Then the concatenation will be finally inputted into the feedforward neural network for the final prediction. The low-level function defines a classifier layer with the input of node embedding list (each element in the list refers to a node embedding tensor at each layer) and the output of legit tensor after prediction. The user can specify the index of edge (represented as tuple of nodes pair indexes) that needs prediction. If not specified, the classifier will be applied to all pair of nodes. 

Below is how the StackedElementProdLayer module constructed.

.. code::

  class StackedElementProdLayer(LinkPredictionLayerBase):   
     def __init__(self, input_size,  hidden_size, num_class, num_channel):        
        super(StackedElementProdLayer, self).__init__() 
            
        #build the linear module list
        self.num_channel=num_channel
        self.ffnn= nn.Linear(num_channel*hidden_size, num_class)


``num_channel`` indicate how many channels of node embedding will be stacked together and are used for the final prediction.

After successfully define the module, we implement the ``forward()`` part to get the prediction results (logins tensor) of specific pair of nodes based on several channels of node embedding, as shown in the below example. Similar to the ``ConcatFeedForwardNNLayer``, we first get the ``src_idx`` and ``dst_idx``. Based on them, the final prediction is conducted as

.. code::

        edge_emb=[]

        for channel_idx in range(self.num_channel):
            edge_emb.append(node_emb[channel_idx][src_idx,:]*node_emb[channel_idx][dst_idx,:])    
            
        return self.ffnn(torch.cat(edge_emb, dim=1))  

In this situation, the ``node_emb`` is not a tensor, but a list of tensor.
 

To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function here, where the input and output are both the graph in type of `GraphData`. The node embedding tensor at channel `N` should be stored in the node feature field named "node_emb_<N>"  in the input_graph for prediction. The computed logit tensor for each pair of nodes in the graph are stored in the edge feature field named "edge_logits". The input graph can be either batched graph or original single graph. The ``forward`` part of ``StackedElementProd`` is the same to that of ``ConcatFeedForwardNN``.




