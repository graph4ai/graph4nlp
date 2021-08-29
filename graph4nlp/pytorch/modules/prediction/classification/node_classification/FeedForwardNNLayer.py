import collections
import torch as th
from torch import nn

from ..base import NodeClassifierLayerBase


class FeedForwardNNLayer(NodeClassifierLayerBase):
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
        super(FeedForwardNNLayer, self).__init__()

        if activation is None:
            activation = nn.ReLU()

        # build the linear module list
        module_seq = []

        for layer_idx in range(len(hidden_size)):
            if layer_idx == 0:
                module_seq.append(
                    ("linear" + str(layer_idx), nn.Linear(input_size, hidden_size[layer_idx]))
                )
            else:
                module_seq.append(
                    (
                        "linear" + str(layer_idx),
                        nn.Linear(hidden_size[layer_idx - 1], self.hidden_size[layer_idx]),
                    )
                )
            module_seq.append(("activate" + str(layer_idx), activation))

        module_seq.append(("linear_end", nn.Linear(hidden_size[-1], num_class)))

        self.classifier = nn.Sequential(collections.OrderedDict(module_seq))

    def forward(self, node_emb, node_idx=None):
        r"""
        Forward functions to compute the logits tensor for node classification.



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
        if node_idx is None:
            return self.classifier(node_emb)
        else:
            new_emb_new = node_emb[th.tensor(node_idx), :]  # get the required node embeddings.
            return self.classifier(new_emb_new)
