from torch import nn
import torch as th
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence
from ..base import NodeClassifierBase
from .BiLSTMFeedForwardNNLayer import BiLSTMFeedForwardNNLayer
from .....data.data import *


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
    def __init__(self, input_size, num_class, hidden_size=None, dropout=0):
        
        super(BiLSTMFeedForwardNN, self).__init__()
        
        self.classifier=BiLSTMFeedForwardNNLayer(input_size, num_class, hidden_size,dropout=dropout)
        self.num_class=num_class

    def forward(self, input_graph):
        r"""
        Forward functions to compute the logits tensor for node classification.
    
      
    
        Parameters
        ----------
    
        input graph : GraphData
                     The tensors stored in the node feature field named "node_emb"  in the 
                     input_graph are used  for classification.
                     GraphData are bacthed and needs to unbatch to each sentence.

    
        Returns 
        ---------
        
        output_graph : GraphData
                      The computed logit tensor for each nodes in the graph are stored
                      in the node feature field named "node_logits".
                      logit tensor shape is: [num_class] 
        """ 
        sent_graph_list=from_batch(input_graph)
        emb_list=[]
        sent_len=[]

        for g in sent_graph_list:
            emb_list.append(g.node_features['node_emb'])
            sent_len.append(g.node_features['node_emb'].size()[0])            
            
        node_emb_padded=pad_sequence(emb_list,batch_first=True) #get the padded node embeddings from a batch of sentence graph         
        bilstm_emb=self.classifier(node_emb_padded) #dimension: batch*max_sent_len*emb_len 

          
        #write back into the sent_graph_list
        unpadded_emb=[]
        for i in range(len(bilstm_emb)):            
           unpadded_emb.append(bilstm_emb[i][:sent_len[i]]) #store the logits tensor into the graph
           
        input_graph.node_features['logits']=th.cat(unpadded_emb)
        return input_graph



