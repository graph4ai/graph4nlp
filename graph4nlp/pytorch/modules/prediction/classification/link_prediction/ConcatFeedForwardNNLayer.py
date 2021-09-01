import torch
from torch import nn

from ..base import LinkPredictionLayerBase


class ConcatFeedForwardNNLayer(LinkPredictionLayerBase):
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
        super(ConcatFeedForwardNNLayer, self).__init__()

        # build the linear module list
        self.activation = activation
        self.ffnn_all1 = nn.Linear(2 * input_size, hidden_size)
        self.ffnn_all2 = nn.Linear(hidden_size, num_class)
        if activation is None:
            activation = nn.ReLU()

    def forward(self, node_emb, edge_idx=None):
        r"""
        Forward functions to compute the logits tensor for node classification.

        Parameters
        ----------

        node_emb : tensor [N,H]
            N: number of nodes
            H: length of the node embeddings
        edge_idx : a list of index of edge (represented as tuple of nodes pair indexes)
        that needs prediction.
            Default: 'None', doing link prediction for all pairs of nodes.
            Example: [(1,2),(1,0),(2,9)]

        Returns
        -------
        logit tensor: [M, num_class] The score logits for all links that need to be preidcted.
            If edge_idx is given, the order of the predicted logits for edges is the same with
            that in the edge_idx
            If full prediction is select (default),the order of predicted logits are like:
                "[(0,0),(0,1),...(0,N),(1,0),(1,1),....(N,N)]"
        """
        if edge_idx is None:
            # get the index list for all the node pairs
            num_node = node_emb.shape[0]
            node_idx_list = list(range(num_node))
            src_idx = torch.tensor(node_idx_list).view(-1, 1).repeat(1, num_node).view(-1)
            dst_idx = torch.tensor(node_idx_list).view(1, -1).repeat(num_node, 1).view(-1)
        else:
            # get the index list for required pairs of nodes
            src_idx = torch.tensor([tuple_idx[0] for tuple_idx in edge_idx])
            dst_idx = torch.tensor([tuple_idx[1] for tuple_idx in edge_idx])

        src_emb = node_emb[src_idx, :]  # input the source node embeddings into ffnn
        dst_emb = node_emb[dst_idx, :]  # input the destinate node embeddings into ffnn
        fused_emb = self.ffnn_all1(torch.cat([src_emb, dst_emb], dim=1))
        return self.ffnn_all2(self.activation(fused_emb))
