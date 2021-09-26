import torch
from torch import nn

from ..base import KGCompletionLayerBase


class DistMultLayer(KGCompletionLayerBase):
    r"""Specific class for knowledge graph completion task.

    DistMult from paper `Embedding entities and relations for learning and
    inference in knowledge bases <https://arxiv.org/pdf/1412.6575.pdf>`__.

    .. math::
        f(s, r, o) & = e_s^T R_r e_o

    Parameters
    ----------
    input_dropout: float
        Dropout for node_emb and rel_emb. Default: 0.0

    loss_name: str
        The loss type selected fot the KG completion task.

    """

    def __init__(self, input_dropout=0.0, loss_name="BCELoss"):
        super(DistMultLayer, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.loss_name = loss_name

    def forward(self, e1_emb, rel_emb, all_node_emb, multi_label=None):
        r"""

        Parameters
        ----------

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
        -------
        pred: tensor [B, N].
            The score logits for all nodes preidcted.

        pred_pos: tensor [B_p]
            The predition scores of positive examples.

        pred_neg: tensor [B_n]
            The predition scores of negative examples.
            B_p + B_n == B * N.

        """
        # dropout
        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        logits = torch.mm(e1_emb * rel_emb, all_node_emb.weight.transpose(1, 0))

        if self.loss_name in ["SoftMarginLoss"]:
            # target labels are numbers selecting from -1 and 1.
            pred = torch.tanh(logits)
        else:
            # target labels are numbers selecting from 0 and 1.
            pred = torch.sigmoid(logits)

        if multi_label is not None:
            idxs_pos = torch.nonzero(multi_label == 1.0)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(multi_label == 0.0)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]
            return pred, pred_pos, pred_neg
        else:
            return pred
