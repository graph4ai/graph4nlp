from __future__ import absolute_import
from torch import nn


class ClassifierBase(nn.Module):
  
    def __init__(self): 
        super(ClassifierBase, self).__init__()

    def forward(self, node_emb):      
        raise NotImplementedError()
        
        
        

class ClassifierLayerBase(nn.Module):
  
    def __init__(self): 
        super(ClassifierLayerBase, self).__init__()

    def forward(self, node_emb):      
        raise NotImplementedError()
        
        
        

class NodeClassifierBase(ClassifierBase):
    
    def __init__(self):
        super(NodeClassifierBase, self).__init__()


    def forward(self, node_emb, node_idx=None):       
        raise NotImplementedError()    
        
        
    
class NodeClassifierLayerBase(ClassifierLayerBase):
    
    def __init__(self):
        super(NodeClassifierLayerBase, self).__init__()


    def forward(self, node_emb, node_idx=None):       
        raise NotImplementedError()    
