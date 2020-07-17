from __future__ import absolute_import
from torch import nn


class ClassifierBase(nn.Module):
    """
    Base class for classification task.

    ...

    Attributes
    ----------

    input_size : int 
                 the length of input node embeddings
    num_class: int 
               the number of classification catrgoriey 

    Methods
    -------

    forward(node_emb)

        Generate the classification logits.         

    """    
    def __init__(self, input_size, num_class):

        super(ClassifierBase, self).__init__()

        self.input_size = input_size
        self.num_class = num_class


    def forward(self, node_emb):      
        raise NotImplementedError()


class NodeClassifierBase(ClassifierBase):
    """
    Base class for node classification task.

    ...

    Attributes
    ----------

    input_size : int 
                 the length of input node embeddings
    num_class: int 
               the number of node catrgoriey for classification

    Methods
    -------

    forward(node_emb)

        Generate the node classification logits.         

    """    
    def __init__(self, input_size, num_class):

        super(NodeClassifierBase, self).__init__(input_size, num_class)


    def forward(self, node_emb, node_idx=None):       
        raise NotImplementedError()    
    
    