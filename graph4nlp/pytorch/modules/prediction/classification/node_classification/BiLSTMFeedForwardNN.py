from ..base import NodeClassifierBase
from .BiLSTMFeedForwardNNLayer import BiLSTMFeedForwardNNLayer


class BiLSTMFeedForwardNN(NodeClassifierBase):
    """
    Specific class for node classification task.

    ...

    Attributes
    ----------

    input_size : int
                 the length of input node embeddings
    num_class: int
               the number of node catrgoriey for classification
    hidden_size: the hidden size of the linear layer

    Methods
    -------

    forward(node_emb)

        Generate the node classification logits.

    """

    def __init__(self, input_size, num_class, hidden_size=None, dropout=0):

        super(BiLSTMFeedForwardNN, self).__init__()

        self.classifier = BiLSTMFeedForwardNNLayer(
            input_size, num_class, hidden_size, dropout=dropout
        )
        self.num_class = num_class

    def forward(self, input_graph):
        r"""
        Forward functions to compute the logits tensor for node classification.



        Parameters
        ----------

        input graph : GraphData
                     The tensors stored in the node feature field named "node_emb"  in the
                     input_graph are used  for classification.
                     GraphData are bacthed and needs to unbatch to each sentence.


        Returns
        ---------

        output_graph : GraphData
                      The computed logit tensor for each nodes in the graph are stored
                      in the node feature field named "node_logits".
                      logit tensor shape is: [num_class]
        """
        if input_graph.batch_size is None:
            node_emb_padded = input_graph.node_features["node_emb"]
            len_emb = node_emb_padded.shape[0]
            bilstm_emb = self.classifier(
                node_emb_padded.reshape([1, len_emb, -1])
            )  # dimension: batch*max_sent_len*emb_len
            input_graph.node_features["logits"] = bilstm_emb.reshape(-1, self.num_class)
        else:
            node_emb_padded = input_graph.batch_node_features["node_emb"]
            bilstm_emb = self.classifier(node_emb_padded)  # dimension: batch*max_sent_len*emb_len
            input_graph.batch_node_features["logits"] = bilstm_emb

        return input_graph
