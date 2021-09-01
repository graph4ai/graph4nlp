import torch

from .StackedElementProdLayer import StackedElementProdLayer


class StackedElementProd(StackedElementProdLayer):
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

    """

    def __init__(self, input_size, hidden_size, num_class):
        super(StackedElementProd, self).__init__()

        self.num_channel = num_class
        self.classifier = StackedElementProdLayer(
            input_size, num_class, self.num_channel, hidden_size
        )

    def forward(self, input_graph):
        r"""
        Forward functions to compute the logits tensor for link prediction.


        Parameters
        ----------

        input graph : GraphData
                     The tensors stored in the node feature field named as
                     "node_emb_"+num_channel (start from "node_emb_0")
                     in the input_graph are used  for link prediction.



        Returns
        ---------

        output_graph : GraphData
                      The computed logit tensor for each pair of nodes in the graph are stored
                      in the node feature field named "edge_logits".
                      logit tensor shape is: [num_class]
        """
        # get the nod embedding from the graph
        node_emb_list = []
        for channel_idx in range(self.num_channel):
            node_emb_list.append(input_graph.node_features["node_emb_" + str(channel_idx)])

        # add the edges and edge prediction logits into the graph
        num_node = node_emb_list[0].shape[1]
        node_idx_list = list(range(num_node))
        src_idx = torch.tensor(node_idx_list).view(-1, 1).repeat(1, num_node).view(-1)
        dst_idx = torch.tensor(node_idx_list).view(1, -1).repeat(num_node, 1).view(-1)
        input_graph.add_edges(src_idx, dst_idx)
        input_graph.edge_features["logits"] = self.classifier(node_emb_list)

        return input_graph
