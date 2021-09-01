import torch
from torch import nn

from ..base import LinkPredictionBase
from .ElementSumLayer import ElementSumLayer


class ElementSum(LinkPredictionBase):
    r"""Specific class for link prediction task.

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

    def __init__(self, input_size, hidden_size, num_class, activation=None):
        super(ElementSum, self).__init__()

        if activation is None:
            activation = nn.ReLU()

        self.classifier = ElementSumLayer(input_size, num_class, hidden_size, activation)

    def forward(self, input_graph):
        r"""
        Forward functions to compute the logits tensor for link prediction.


        Parameters
        ----------

        input graph : GraphData
                     The tensors stored in the node feature field named "node_emb"  in the
                     input_graph are used  for link prediction.


        Returns
        ---------

        output_graph : GraphData
                      The computed logit tensor for each pair of nodes in the graph are stored
                      in the node feature field named "edge_logits".
                      logit tensor shape is: [num_class]
        """
        # get the nod embedding from the graph
        node_emb = input_graph.node_features["node_emb"]

        # add the edges and edge prediction logits into the graph
        num_node = node_emb.shape[0]
        node_idx_list = list(range(num_node))
        src_idx = torch.tensor(node_idx_list).view(-1, 1).repeat(1, num_node).view(-1)
        dst_idx = torch.tensor(node_idx_list).view(1, -1).repeat(num_node, 1).view(-1)
        input_graph.add_edges(src_idx, dst_idx)
        input_graph.edge_features["logits"] = self.classifier(node_emb)

        return input_graph
