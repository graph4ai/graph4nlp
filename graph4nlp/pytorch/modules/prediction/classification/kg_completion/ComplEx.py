from .....data.data import GraphData
from ..base import KGCompletionBase
from .ComplExLayer import ComplExLayer


class ComplEx(KGCompletionBase):
    r"""Specific class for knowledge graph completion task.

    ComplEx from paper `Complex Embeddings for Simple Link Prediction
    <http://proceedings.mlr.press/v48/trouillon16.pdf>`__.


    Parameters
    ----------
    input_dropout: float
        Dropout for node_emb and rel_emb. Default: `0.0`

    loss_name: str
        The loss type selected fot the KG completion task. Default: `'BCELoss'`
    """

    def __init__(self, input_dropout=0.0, loss_name="BCELoss"):
        super(ComplEx, self).__init__()
        self.loss_name = loss_name
        self.classifier = ComplExLayer(input_dropout, loss_name)

    def forward(
        self,
        input_graph: GraphData,
        e1_embedded_real,
        rel_embedded_real,
        e1_embedded_img,
        rel_embedded_img,
        all_node_emb_real,
        all_node_emb_img,
        multi_label=None,
    ):
        r"""
        Forward functions to compute the logits tensor for kg completion.

        Parameters
        ----------

        input graph : GraphData
            The tensors stored in the node feature field named "node_emb" and
            "rel_emb" in the input_graph are used for knowledge graph completion.

        e1_embedded_real : tensor [B, H]
            The selected entity_1 real embeddings of a batch.
            B: batch size
            H: length of the node embeddings (entity embeddings)

        rel_embedded_real : tensor [B, H]
            The selected relation real embeddings of a batch.
            B: batch size
            H: length of the edge embeddings (relation embeddings)

        e1_embedded_img : tensor [B, H]
            The selected entity_1 img embeddings of a batch.
            B: batch size
            H: length of the node embeddings (entity embeddings)

        rel_embedded_img : tensor [B, H]
            The selected relation img embeddings of a batch.
            B: batch size
            H: length of the edge embeddings (relation embeddings)

        all_node_emb_real :  torch.nn.modules.sparse.Embedding [N, H]
            All node real embeddings.
            N: number of nodes in the whole KG graph
            H: length of the node real embeddings (entity embeddings)

        all_node_emb_img :  torch.nn.modules.sparse.Embedding [N, H]
            All node img embeddings.
            N: number of nodes in the whole KG graph
            H: length of the node img embeddings (entity embeddings)

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
                e1_embedded_real,
                e1_embedded_img,
                rel_embedded_real,
                rel_embedded_img,
                all_node_emb_real,
                all_node_emb_img,
            )
        else:
            (
                input_graph.graph_attributes["logits"],
                input_graph.graph_attributes["p_score"],
                input_graph.graph_attributes["n_score"],
            ) = self.classifier(
                e1_embedded_real,
                e1_embedded_img,
                rel_embedded_real,
                rel_embedded_img,
                all_node_emb_real,
                all_node_emb_img,
                multi_label,
            )

        return input_graph
