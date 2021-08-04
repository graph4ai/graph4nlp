.. _guide-node_classification:

Node Classification
===================

Node classification is a downstream task that are normally observed in the GNN-based NLP tasks, such as sequence labeling and name entity recognition. The process is about classify the label of the each node in the graph based on the node embeddings that learnt from the GNNs modules.

To facilitate the implementation of node classification task, we provide both high-level and low-level APIs to users to easily define a multi-layer node classification function. Besides, for each level's APIs, we support two popularly used node classifiers, one is based on the BiLSTM and feedforward neural network (BiLSTMFeedForwardNN), another one is based on simple feedforward neural network (FeedForwardNN).



BiLSTMFeedForwardNNLayer
-------------------

This function is based on a combination of the BiLSTM layer and a feedforward layer. The low-level function defines the a single layer classifier with the input of node embedding tensor and the output of legit tensor after classification. 

Below is an example to construct the BiLSTMFeedForwardNNlayer module. If ``hidden_size`` is not ``None``, we add an additional hidden layer after the last. Else, we directly use the output of the lstm as the classification logits.

.. code::
 
   from torch import nn
   import torch as th

   class BiLSTMFeedForwardNNLayer(NodeClassifierLayerBase):   
     def __init__(self, input_size, output_size, hidden_size= None, dropout=0):
        super(BiLSTMFeedForwardNNLayer, self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size

        if hidden_size!=None:
           self.lstm = nn.LSTM(input_size, 
                               self.hidden_size//2, 
                               num_layers=1,
                               bidirectional=True,
                               batch_first=True,
                               dropout=dropout)           
           self.linear = nn.Linear(hidden_size, self.output_size)
        else:
           self.lstm = nn.LSTM(input_size,         
      			       self.output_size//2, 
      			       num_layers=1, 
      			       bidirectional=True,
      			       batch_first=True,
      			       dropout=dropout) 

After the construction of the module, the next step is to make the ``forward()`` part. The user can specify the index of nodes that needs to be classified in ``node_idx``. If not specified, the classifier will be applied to all the nodes. 

.. code::

	def forward(self, node_emb, node_idx=None):

     	   if node_idx:
               node_emb = node_emb[th.tensor(node_idx), :]   
       	 	 
  	   batch_size=node_emb.size(0) 
 
           self.hidden = self.init_hidden(batch_size)  
           lstm_out, self.hidden = self.lstm(node_emb, self.hidden)

           if  self.hidden_size is None:                                      
              return lstm_out  
           else:
              lstm_feats = self.linear(lstm_out)


The final output of this module is the classification logits.



BiLSTMFeedForwardNN
-------------------


To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for node classification, where the input and output are both the graph in type of `GraphData`. 

Below is an example to construct the BiLSTMFeedForwardNN module.

.. code::
    class BiLSTMFeedForwardNN(NodeClassifierBase):   
       def __init__(self, input_size, num_class, hidden_size=None, dropout=0):
        
        super(BiLSTMFeedForwardNN, self).__init__()
        
        self.classifier=BiLSTMFeedForwardNNLayer(input_size, num_class, hidden_size,dropout=dropout)
        self.num_class=num_class

The ``num_class`` defines the the number of node categoriey for classification. 


After define the module, we implement the``forward()`` part. In the ``forward()`` part, the node embedding tensor should be stored in the node feature field named ``node_emb``  in the input_graph for classification. If the input graph is a ``batched graphs``, we can get the node embedding and do the classification as

.. code::

    node_emb_padded=input_graph.node_features['node_emb']
    len_emb=node_emb_padded.shape[0]
    bilstm_emb=self.classifier(node_emb_padded.reshape([1,len_emb,-1])) 

If the input graph is an origin graph, the operation is as

.. code::

    node_emb_padded=input_graph.batch_node_features['node_emb']
    bilstm_emb=self.classifier(node_emb_padded)  


After getting the ``bilstm_emb``, which is also the computed logit tensor for each nodes in the graph, they are stored in the node feature field named ``node_logit`` as

.. code-block:: python

    input_graph.batch_node_features['logits']=bilstm_emb





FeedForwardNN
---------------

This function is based on a combination of several feedforward layer. The low-level function defines the classifier layers with the input of node embedding tensor and the output of legit tensor after classification. 

Below is an example to construct the FeedForwardNNLayer module.

.. code::

	class FeedForwardNNLayer(NodeClassifierLayerBase):
  	    def __init__(self, input_size, num_class, hidden_size,activation=nn.ReLU()):        
      		  super(FeedForwardNNLayer, self).__init__()
           
            
      		  #build the linear module list
       		  module_seq=[]        
        
       		  for layer_idx in range(len(hidden_size)):

       		     if layer_idx==0:
		          module_seq.append(('linear'+str(layer_idx),nn.Linear(input_size,hidden_size[layer_idx])))
            	     else:
                          module_seq.append(('linear'+str(layer_idx),nn.Linear(hidden_size[layer_idx-1],self.hidden_size[layer_idx])))

                     module_seq.append(('activate'+str(layer_idx),activation))
            
                  module_seq.append(('linear_end',nn.Linear(hidden_size[-1],num_class)))
        
                  self.classifier = nn.Sequential(collections.OrderedDict(module_seq))


``hidden_size`` can be a list of int values. Each element in ``hidden_size`` is the size of each hidden layer. 

After construct the module, the next step is to make the ``forward()`` part. The user can specify the index of nodes that needs to be classified in ``node_idx``. If not specified, the classifier will be applied to all the nodes. The output of this module is the node embedding tensor.

.. code::

    def forward(self, node_emb, node_idx=None):
 
        if node_idx == None:
            return self.classifier(node_emb)
        else:
            new_emb_new = node_emb[th.tensor(node_idx), :]  # get the required node embeddings.

            return self.classifier(new_emb_new)
   
To facilitate the easily implementation of the pipeline of GNN-based NLP application, we also provide the high-level function for multi-layer node classification, where the input and output are both the graph in type of `GraphData`. The node embedding tensor should be stored in the node feature field named "node_emb"  in the input_graph for classification. The computed logit tensor for each nodes in the graph are stored in the node feature field named "node_logits". The input graph can be either batched graph or original single graph. Below is an example to define and call the FeedForwardNN API.


Below is an example to construct the FeedForwardNN module. 

.. code::

  class FeedForwardNN(NodeClassifierBase):
     def __init__(self, 
                 input_size, 
                 num_class, 
                 hidden_size, 
                 activation=nn.ReLU()):        
        super(FeedForwardNN, self).__init__()

        self.classifier=FeedForwardNNLayer(input_size, num_class, hidden_size, activation)


After the construction of the module, the next step is to make the ``forward()`` part. The user can specify the index of nodes that needs to be classified in ``node_idx``. If not specified, the classifier will be applied to all the nodes. After getting the ``node_emb``, which is also the computed logit tensor for each nodes in the graph, they are stored in the node feature field named ``node_logit`` and the output is the GraphData.

.. code::

  def forward(self, input_graph):
        node_emb=input_graph.node_features['node_emb']
        input_graph.node_features['logits']=self.classifier(node_emb)
        return input_graph

