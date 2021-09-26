import collections
from torch import nn

from ..base import GraphClassifierBase, GraphClassifierLayerBase
from .avg_pooling import AvgPooling
from .max_pooling import MaxPooling


class FeedForwardNN(GraphClassifierBase):
    r"""FeedForwardNN class for graph classification task.

    Parameters
    ----------
    input_size : int
        The dimension of input graph embeddings.
    num_class : int
        The number of classes for classification.
    hidden_size : list of int
        Hidden size per NN layer.
    activation: nn.Module, optional
        The activation function, default: `nn.ReLU()`.
    """

    def __init__(
        self,
        input_size,
        num_class,
        hidden_size,
        activation=None,
        graph_pool_type="max_pool",
        **kwargs
    ):
        super(FeedForwardNN, self).__init__()

        if not activation:
            activation = nn.ReLU()

        if graph_pool_type == "avg_pool":
            self.graph_pool = AvgPooling()
        elif graph_pool_type == "max_pool":
            self.graph_pool = MaxPooling(**kwargs)
        else:
            raise RuntimeError("Unknown graph pooling type: {}".format(graph_pool_type))

        self.classifier = FeedForwardNNLayer(input_size, num_class, hidden_size, activation)

    def forward(self, graph):
        r"""Compute the logits tensor for graph classification.

        Parameters
        ----------
        graph : GraphData
            The graph data containing graph embeddings.

        Returns
        -------
        list of GraphData
            The output graph data containing logits tensor for graph classification.
        """
        graph_emb = self.graph_pool(graph, "node_emb")
        logits = self.classifier(graph_emb)
        graph.graph_attributes["logits"] = logits

        return graph


class FeedForwardNNLayer(GraphClassifierLayerBase):
    r"""FeedForwardNNLayer class for graph classification task.

    Parameters
    ----------
    input_size : int
        The dimension of input graph embeddings.
    num_class : int
        The number of classes for classification.
    hidden_size : list of int
        Hidden size per NN layer.
    activation: nn.Module, optional
        The activation function, default: `nn.ReLU()`.
    """

    def __init__(self, input_size, num_class, hidden_size, activation=None):
        super(FeedForwardNNLayer, self).__init__()

        if not activation:
            activation = nn.ReLU()

        # build the linear module list
        module_seq = []

        for layer_idx in range(len(hidden_size)):
            if layer_idx == 0:
                module_seq.append(
                    ("fc" + str(layer_idx), nn.Linear(input_size, hidden_size[layer_idx]))
                )
            else:
                module_seq.append(
                    (
                        "fc" + str(layer_idx),
                        nn.Linear(hidden_size[layer_idx - 1], self.hidden_size[layer_idx]),
                    )
                )

            module_seq.append(("activate" + str(layer_idx), activation))

        module_seq.append(("fc_end", nn.Linear(hidden_size[-1], num_class)))

        self.classifier = nn.Sequential(collections.OrderedDict(module_seq))

    def forward(self, graph_emb):
        r"""Compute the logits tensor for graph classification.

        Parameters
        ----------
        graph_emb : torch.Tensor
            The input graph embeddings.

        Returns
        -------
        torch.Tensor
            The output logits tensor for graph classification.
        """
        return self.classifier(graph_emb)
