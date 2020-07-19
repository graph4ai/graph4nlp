from torch import nn
import torch as th
import torch.autograd as autograd
from ..base import NodeClassifierBase
from BiLSTMFeedForwardNNLayer import BiLSTMFeedForwardNNLayer

class BiLSTMFeedForwardNN(NodeClassifierBase):
    """
    Specific class for node classification task.

    ...

    Attributes
    ----------

    input_size : int 
                 the length of input node embeddings
    num_class: int 
               the number of node catrgoriey for classification
    hidden_size: the hidden size of the linear layer

    Methods
    -------

    forward(node_emb)

        Generate the node classification logits.         

    """     
    def __init__(self, input_size, num_class, hidden_size):
        
        super(BiLSTMFeedForwardNN, self).__init__()
        
        self.classifier=BiLSTMFeedForwardNNLayer(input_size, num_class, hidden_size)


    def forward(self, input_graph):
        r"""
        Forward functions to compute the logits tensor for node classification.
    
      
    
        Parameters
        ----------
    
        input graph : GraphData
                     The tensors stored in the node feature field named "node_emb"  in the 
                     input_graph are used  for classification.

    
        Returns 
        ---------
        
        output_graph : GraphData
                      The computed logit tensor for each nodes in the graph are stored
                      in the node feature field named "node_logits".
                      logit tensor shape is: [num_class] 
        """ 
        node_emb=input_graph.ndata['node_emb'] #get the node embeddings from the graph
         
        input_graph.ndata['logits']=self.classifier(lstm_feats) #store the logits tensor into the graph
        
        return input_graph



