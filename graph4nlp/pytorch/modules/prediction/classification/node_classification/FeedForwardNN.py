from torch import nn

from ..base import NodeClassifierBase
from .FeedForwardNNLayer import FeedForwardNNLayer


class FeedForwardNN(NodeClassifierBase):
    r"""Specific class for node classification task.


    Parameters
    ----------

    input_size : int
                 The length of input node embeddings
    num_class : int
               The number of node catrgoriey for classification
    hidden_size : list of int type values
                  Example for two layers's FeedforwardNN: [50, 20]
    activation: the activation function class for each fully connected layer
                Default: nn.ReLU()
                Example: nn.ReLU(),nn.Sigmoid().

    """

    def __init__(self, input_size, num_class, hidden_size, activation=None):
        super(FeedForwardNN, self).__init__()

        if activation is None:
            activation = nn.ReLU()
        self.classifier = FeedForwardNNLayer(input_size, num_class, hidden_size, activation)

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

        node_emb = input_graph.node_features["node_emb"]
        input_graph.node_features["logits"] = self.classifier(node_emb)

        return input_graph
