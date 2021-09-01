import torch
from torch import nn

from ..base import LinkPredictionLayerBase


class StackedElementProdLayer(LinkPredictionLayerBase):
    r"""Specific class for link prediction task.


    Parameters
    ----------

    input_size : int
                 The length of input node embeddings
    num_class : int
               The number of node catrgoriey for classification
    num_channel: int
               The number of channels for node embeddings to be used for link prediction
    hidden_size : list of int type values
                  Example for two layers's FeedforwardNN: [50, 20]
    activation: the activation function class for each fully connected layer
                Default: nn.ReLU()
                Example: nn.ReLU(),nn.Sigmoid().

    """

    def __init__(self, input_size, hidden_size, num_class, num_channel):
        super(StackedElementProdLayer, self).__init__()

        # build the linear module list
        self.num_channel = num_channel
        self.ffnn = nn.Linear(num_channel * hidden_size, num_class)

    def forward(self, node_emb, edge_idx=None):
        r"""
        Forward functions to compute the logits tensor for link classification.
        This method requires the input of node embeddings generated from each
        layer of the node embedding process.
        Parameters
        ----------

        node_emb : list of tensor. Each tensor ([N,H]) refers to node mebdding generated
                   from one layer of node embedding process.
            N: number of nodes
            H: length of the node embeddings
        edge_idx : a list of index of edge (represented as tuple of nodes pair indexes) that
        needs prediction.
            Default: 'None', doing link prediction for all pairs of nodes.
            Example: [(1,2),(1,0),(2,9)]

        Returns
        -------
        logit tensor: [M, num_class] The score logits for all links that need to be preidcted.
            If edge_idx is given, the order of the predicted logits for edges is the same with that
            in the edge_idx
            If full prediction is select (default),the order of predicted logits are like:
                "[(0,0),(0,1),...(0,N),(1,0),(1,1),....(N,N)]"
        """
        if edge_idx is None:
            # get the index list for all the node pairs
            num_node = node_emb[0].shape[0]
            node_idx_list = [range(num_node)]
            src_idx = torch.tensor(node_idx_list).view(-1, 1).repeat(1, num_node).view(-1)
            dst_idx = torch.tensor(node_idx_list).view(1, -1).repeat(num_node, 1).view(-1)
        else:
            # get the index list for required pairs of nodes
            src_idx = torch.tensor([tuple_idx[0] for tuple_idx in edge_idx])
            dst_idx = torch.tensor([tuple_idx[1] for tuple_idx in edge_idx])

        edge_emb = []
        for channel_idx in range(self.num_channel):
            edge_emb.append(node_emb[channel_idx][src_idx, :] * node_emb[channel_idx][dst_idx, :])

        return self.ffnn(torch.cat(edge_emb, dim=1))
