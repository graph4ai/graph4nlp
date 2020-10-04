from torch import nn
import torch as th
import torch.autograd as autograd
from ..base import NodeClassifierLayerBase

class BiLSTMFeedForwardNNLayer(NodeClassifierLayerBase):
    r"""Specific class for node classification layer.


    Parameters
    ----------

    input_size : int 
                 The length of input node embeddings
    num_class : int 
               The number of node catrgoriey for classification
    hidden_size : int
                  the hidden size of the linear layer 

    """       
    def __init__(self, input_size, num_class, hidden_size):
        super(BiLSTMFeedForwardNNLayer, self).__init__()
        self.hidden_size=hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size//2, num_layers=1, bidirectional=True) # Maps the output of the LSTM into tag space.        
        self.linear = nn.Linear(hidden_size, num_class) 
        self.hidden = self.init_hidden()
        
        
    def init_hidden(self):        
        return (autograd.Variable(th.randn(2, 1, self.hidden_size // 2)),                
                autograd.Variable(th.randn(2, 1, self.hidden_size // 2)))


    def forward(self, node_emb, node_idx=None):
        r"""
        Forward functions for classification task.
    
        ...
    
        Parameters
        ----------
    
        node_emb : tensor [N,H]  
                   N: number of nodes    
                   H: length of the node embeddings
        node_idx : a list of index of nodes that needs classification.
                   Default: 'None'
                   Example: [1,3,5]
    
        Returns 
        -------
             logit tensor: [N, num_class] The score logits for all nodes preidcted.
        """ 
        if node_idx:
            node_emb = node_emb[th.tensor(node_idx), :]  # get the required node embeddings.
        self.hidden = self.init_hidden()
        node_emb = node_emb.unsqueeze(1)         
        lstm_out, self.hidden = self.lstm(node_emb, self.hidden)         
        lstm_out = lstm_out.view(-1, self.hidden_size)          
        lstm_feats = self.linear(lstm_out)
           
        return lstm_feats
