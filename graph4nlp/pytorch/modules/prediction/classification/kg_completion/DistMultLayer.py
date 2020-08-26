from torch import nn
import torch
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

    rel_emb_from_gnn: bool
        If `rel_emb` is computed from GNN, rel_emb_from_gnn is set to `True`.
        Else, rel_emb is initialized as nn.Embedding randomly. Default: `True`.

    num_relations: int
        Number of relations. `num_relations` is needed if rel_emb_from_gnn==True.
        Default: `None`.

    embedding_dim: int
        Dimension of the rel_emb. `embedding_dim` is needed if rel_emb_from_gnn==True.
        Default: `0`.

    loss_name: str
        The loss type selected fot the KG completion task.

    """

    def __init__(self,
                 input_dropout=0.0,
                 rel_emb_from_gnn=True,
                 num_relations=None,
                 embedding_dim=None,
                 loss_name='BCELoss'):
        super(DistMultLayer, self).__init__()
        self.rel_emb_from_gnn = rel_emb_from_gnn
        self.inp_drop = nn.Dropout(input_dropout)
        if self.rel_emb_from_gnn == False:
            assert num_relations != None
            assert embedding_dim != None
            self.rel_emb = nn.Embedding(num_relations, embedding_dim)
            # self.reset_parameters()
        self.loss_name = loss_name
        self.reset_parameters()


    def reset_parameters(self):
        if self.rel_emb_from_gnn == False:
            nn.init.xavier_normal_(self.rel_emb.weight.data)


    def forward(self,
                node_emb,
                rel_emb=None,
                list_e_r_pair_idx=None,
                list_e_e_pair_idx=None,
                multi_label=None):
        r"""

        Parameters
        ----------

        node_emb: tensor [N,H]
            N: number of nodes in the whole KG graph
            H: length of the node embeddings (entity embeddings)

        rel_emb: tensor [N_r,H]
            N_r: number of relations in the whole KG graph
            H: length of the relation embeddings

        list_e_r_pair_idx: list of tuple
            a list of index of head entities and relations that needs
            predicting the tail entities between them. Default: `None`

        list_e_e_pair_idx: list of tuple
            a list of index of head entities and tail entities that needs
            predicting the relations between them. Default: `None`.
            Only one of `list_e_r_pair_idx` and `list_e_e_pair_idx` can be `None`.

        multi_label: tensor [L, N]
            multi_label is a binary matrix. Each element can be equal to 1 for true label
            and 0 for false label (or 1 for true label, -1 for false label).
            multi_label[i] represents a multi-label of a given head-rel pair or head-tail pair.
            L is the length of list_e_r_pair_idx, list_e_e_pair_idx or batch size.
            N: number of nodes in the whole KG graph.

        Returns
        -------
        pred: tensor [L, N]
            logit tensor. [L, N] The score logits for all nodes preidcted.

        pred_pos: tensor [L_p]
            The predition scores of positive examples.

        pred_neg: tensor [L_n]
            The predition scores of negative examples. L_p + L_n == L * N.

        """
        if self.rel_emb_from_gnn == False:
            assert rel_emb is None
            rel_emb = self.rel_emb.weight

        if list_e_r_pair_idx is None and list_e_e_pair_idx is None:
            raise RuntimeError("Only one of `list_e_r_pair_idx` and `list_e_e_pair_idx` can be `None`.")

        assert node_emb.size()[1]==rel_emb.size()[1]

        if list_e_r_pair_idx is not None:
            ent_idxs = torch.LongTensor([x[0] for x in list_e_r_pair_idx])
            rel_idxs = torch.LongTensor([x[1] for x in list_e_r_pair_idx])

            selected_ent_embs = node_emb[ent_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx
            selected_rel_embs = rel_emb[rel_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx

            # dropout
            selected_ent_embs = self.inp_drop(selected_ent_embs)
            selected_rel_embs = self.inp_drop(selected_rel_embs)

            logits = torch.mm(selected_ent_embs * selected_rel_embs,
                              node_emb.transpose(1, 0))
        elif list_e_e_pair_idx is not None:
            ent_head_idxs = torch.LongTensor([x[0] for x in list_e_e_pair_idx])
            ent_tail_idxs = torch.LongTensor([x[1] for x in list_e_e_pair_idx])

            selected_ent_head_embs = node_emb[ent_head_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx
            selected_ent_tail_embs = rel_emb[ent_tail_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx

            # dropout
            selected_ent_head_embs = self.inp_drop(selected_ent_head_embs)
            selected_ent_tail_embs = self.inp_drop(selected_ent_tail_embs)

            logits = torch.mm(selected_ent_head_embs*selected_ent_tail_embs,
                              rel_emb.transpose(1, 0))

        if self.loss_name in ['SoftMarginLoss']:
            # target labels are numbers selecting from -1 and 1.
            pred = torch.tanh(logits)
        else:
            # target labels are numbers selecting from 0 and 1.
            pred = torch.sigmoid(logits)

        if multi_label is not None:
            idxs_pos = torch.nonzero(multi_label == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(multi_label == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]
            return pred, pred_pos, pred_neg
        else:
            return pred