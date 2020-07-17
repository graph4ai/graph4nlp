from ..base import NodeClassifierBase
import collections
from torch import nn
import torch as th


class FeedForwardNN(NodeClassifierBase):
    """
    Specific class for node classification task.

    ...

    Attributes
    ----------

    input_size : int 
                 the length of input node embeddings
    num_class: int 
               the number of node catrgoriey for classification
    hidden_size: list of int type values
                [hidden_size1, hidden_size2,...]

    Methods
    -------

    forward(node_emb)

        Generate the node classification logits.         

    """     
    def __init__(self, input_size, num_class, hidden_size):        
        super(FeedForwardNN, self).__init__(input_size, num_class)
        self.hidden_size=hidden_size
        self.num_hidden_layers=len(self.hidden_size)
        
        #build the linear module list
        module_seq=[]
        
        for layer_idx in range(self.num_hidden_layers):
            if layer_idx==0:
                module_seq.append(('linear'+str(layer_idx),nn.Linear(self.input_size,self.hidden_size[layer_idx])))
            else:
                module_seq.append(('linear'+str(layer_idx),nn.Linear(self.hidden_size[layer_idx-1],self.hidden_size[layer_idx])))
            module_seq.append(('relu'+str(layer_idx),nn.ReLU()))
            
        module_seq.append(('linear_end',nn.Linear(self.hidden_size[-1],num_class)))
        
        self.classifier = nn.Sequential(collections.OrderedDict(module_seq))

    def forward(self, node_emb, node_idx=None):
        """
        Forward functions for classification task.
    
        ...
    
        Parameters
        ----------
    
        node_emb : tensor [N,H]  
                   N: number of nodes    
                   H: length of the node embeddings
        node_idx : a list of index of nodes that needs classification.
                   Default: 'None'
    
        Returns 
        -------
             logit tensor: [N, num_class] The score logits for all nodes preidcted.
        """ 
        if node_idx == None:
            return self.classifier(node_emb)
        else:
            new_emb_new = node_emb[th.tensor(node_idx), :]  # get the required node embeddings.
            return self.classifier(new_emb_new)

if __name__=="__main__":        
   print('hello')