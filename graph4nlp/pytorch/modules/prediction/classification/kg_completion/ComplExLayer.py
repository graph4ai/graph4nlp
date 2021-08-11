import torch
from torch import nn

from ..base import KGCompletionLayerBase


class ComplExLayer(KGCompletionLayerBase):
    r"""Specific class for knowledge graph completion task.

    ComplEx from paper `Complex Embeddings for Simple Link Prediction
    <http://proceedings.mlr.press/v48/trouillon16.pdf>`__.

    Parameters
    ----------
    input_dropout: float
       Dropout for node_emb and rel_emb. Default: 0.0

    loss_name: str
       The loss type selected fot the KG completion task.

    """

    def __init__(self, input_dropout=0.0, loss_name="BCELoss"):
        super(ComplExLayer, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.loss_name = loss_name

    def forward(
        self,
        e1_embedded_real,
        e1_embedded_img,
        rel_embedded_real,
        rel_embedded_img,
        all_node_emb_real,
        all_node_emb_img,
        multi_label=None,
    ):
        r"""

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
        -------
        pred: tensor [B, N].
            The score logits for all nodes preidcted.

        pred_pos: tensor [B_p]
            The predition scores of positive examples.

        pred_neg: tensor [B_n]
            The predition scores of negative examples.
            B_p + B_n == B * N.

        """
        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(
            e1_embedded_real * rel_embedded_real, all_node_emb_real.weight.transpose(1, 0)
        )
        realimgimg = torch.mm(
            e1_embedded_real * rel_embedded_img, all_node_emb_img.weight.transpose(1, 0)
        )
        imgrealimg = torch.mm(
            e1_embedded_img * rel_embedded_real, all_node_emb_img.weight.transpose(1, 0)
        )
        imgimgreal = torch.mm(
            e1_embedded_img * rel_embedded_img, all_node_emb_real.weight.transpose(1, 0)
        )
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal

        if self.loss_name in ["SoftMarginLoss"]:
            pred = torch.tanh(pred)
        else:
            pred = torch.sigmoid(pred)

        if multi_label is not None:
            idxs_pos = torch.nonzero(multi_label == 1.0)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(multi_label == 0.0)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]

            return pred, pred_pos, pred_neg
        else:
            return pred
