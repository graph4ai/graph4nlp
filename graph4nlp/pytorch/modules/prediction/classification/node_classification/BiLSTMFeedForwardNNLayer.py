import torch as th
import torch.autograd as autograd
from torch import nn

from ..base import NodeClassifierLayerBase


class BiLSTMFeedForwardNNLayer(NodeClassifierLayerBase):
    r"""Specific class for node classification layer.


    Parameters
    ----------

    input_size : int
                 The length of input node embeddings
    output_size : int
               The number of node catrgoriey for classification
    hidden_size : int
                  the hidden size of the linear layer

    """

    def __init__(self, input_size, output_size, hidden_size=None, dropout=0):
        super(BiLSTMFeedForwardNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        if hidden_size is not None:
            self.lstm = nn.LSTM(
                input_size,
                self.hidden_size // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )  # Maps the output of the LSTM into tag space.
            self.linear = nn.Linear(hidden_size, self.output_size)

        else:
            self.lstm = nn.LSTM(
                input_size,
                self.output_size // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )  # Maps the output of the LSTM into tag space.

    def init_hidden(self, batch_size):
        if self.hidden_size is not None:
            return (
                autograd.Variable(th.randn(2, batch_size, self.hidden_size // 2)).to(self.device),
                autograd.Variable(th.randn(2, batch_size, self.hidden_size // 2)).to(self.device),
            )
        else:
            return (
                autograd.Variable(th.randn(2, batch_size, self.output_size // 2)).to(self.device),
                autograd.Variable(th.randn(2, batch_size, self.output_size // 2)).to(self.device),
            )

    def forward(self, node_emb, node_idx=None):
        r"""
        Forward functions for classification task.

        ...

        Parameters
        ----------

        node_emb : padded tensor [B,N,H]
                   B: batch size
                   N: max number of nodes
                   H: length of the node embeddings
        node_idx : a list of index of nodes that needs classification.
                   Default: 'None'
                   Example: [1,3,5]

        Returns
        -------
             logit tensor: [B,N, num_class] The score logits for all nodes preidcted.
        """
        if node_idx:
            node_emb = node_emb[th.tensor(node_idx), :]  # get the required node embeddings.
        batch_size = node_emb.size(0)

        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(node_emb, self.hidden)
        if self.hidden_size is None:
            return lstm_out
        else:
            lstm_feats = self.linear(lstm_out)
            return lstm_feats
