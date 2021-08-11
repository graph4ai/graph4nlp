from .....data.data import GraphData
from ..base import KGCompletionBase
from .DistMultLayer import DistMultLayer


class DistMult(KGCompletionBase):
    r"""Specific class for knowledge graph completion task.

    DistMult from paper `Embedding entities and relations for learning and
    inference in knowledge bases <https://arxiv.org/pdf/1412.6575.pdf>`__.

    Parameters
    ----------
    input_dropout: float
        Dropout for node_emb and rel_emb. Default: 0.0

    loss_name: str
        The loss type selected fot the KG completion task. Default: `'BCELoss'`
    """

    def __init__(self, input_dropout=0.0, loss_name="BCELoss"):
        super(DistMult, self).__init__()
        self.loss_name = loss_name
        self.classifier = DistMultLayer(input_dropout, loss_name)

    def forward(self, input_graph: GraphData, e1_emb, rel_emb, all_node_emb, multi_label=None):
        r"""
        Forward functions to compute the logits tensor for kg completion.

        Parameters
        ----------

        input graph : GraphData
            The tensors stored in the node feature field named "node_emb" and
            "rel_emb" in the input_graph are used for knowledge graph completion.

        e1_emb : tensor [B, H]
            The selected entity_1 embeddings of a batch.
            B: batch size
            H: length of the node embeddings (entity embeddings)

        rel_emb : tensor [B, H]
            The selected relation embeddings of a batch.
            B: batch size
            H: length of the edge embeddings (relation embeddings)

        all_node_emb :  torch.nn.modules.sparse.Embedding [N, H]
            All node embeddings.
            N: number of nodes in the whole KG graph
            H: length of the node embeddings (entity embeddings)

        multi_label: tensor [B, N]
            multi_label is a binary matrix. Each element can be equal to 1 for true label
            and 0 for false label (or 1 for true label, -1 for false label).
            multi_label[i] represents a multi-label of a given head-rel pair.
            B is the batch size.
            N: number of nodes in the whole KG graph.

        Returns
        ---------

        output_graph : GraphData
                      The computed logit tensor for each nodes in the graph are stored
                      in the node feature field named "node_logits".
                      logit tensor shape is: [num_class]
        """

        if multi_label is None:
            input_graph.graph_attributes["logits"] = self.classifier(
                e1_emb, rel_emb, all_node_emb
            )  # [B, N]
        else:
            (
                input_graph.graph_attributes["logits"],
                input_graph.graph_attributes["p_score"],
                input_graph.graph_attributes["n_score"],
            ) = self.classifier(e1_emb, rel_emb, all_node_emb, multi_label)
            # input_graph.graph_attributes['p_score']: [L_p]
            # input_graph.graph_attributes['n_score']: [L_n]
            # L_p + L_n == B * N

        return input_graph
