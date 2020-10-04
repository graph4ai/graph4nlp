from torch import nn
import torch
from ..base import KGCompletionLayerBase
import torch.nn.functional as F


class TransELayer(KGCompletionLayerBase):
    r"""Specific class for knowledge graph completion task.
    TransE from paper `Translating Embeddings for Modeling
    Multi-relational Data <https://papers.nips.cc/paper/5071
    -translating-embeddings-for-modeling-multi-relational-data.pdf>`__.

    .. math::
        f(s, r, o) & = ||e_s + w_r - e_o||_p

    Parameters
    ----------
    p_norm: int
        Default: 1

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
                 p_norm=1,
                 rel_emb_from_gnn=True,
                 num_relations=None,
                 embedding_dim=None,
                 loss_name='BCELoss'):
        super(TransELayer, self).__init__()
        self.p_norm = p_norm
        self.rel_emb_from_gnn = rel_emb_from_gnn
        if self.rel_emb_from_gnn == False:
            assert num_relations != None
            assert embedding_dim != None
            self.rel_emb = nn.Embedding(num_relations, embedding_dim)
            self.reset_parameters()
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
            N: number of relations in the whole KG graph
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

        assert node_emb.size()[1] == rel_emb.size()[1]

        if list_e_r_pair_idx is not None:
            ent_idxs = torch.LongTensor([x[0] for x in list_e_r_pair_idx])
            rel_idxs = torch.LongTensor([x[1] for x in list_e_r_pair_idx])

            selected_ent_embs = node_emb[ent_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx
            selected_rel_embs = rel_emb[rel_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx

            selected_ent_embs = F.normalize(selected_ent_embs, 2, -1)
            selected_rel_embs = F.normalize(selected_rel_embs, 2, -1)
            node_emb = F.normalize(node_emb, 2, -1)

            head_add_rel = selected_ent_embs + selected_rel_embs  # [L, H]
            head_add_rel = head_add_rel.view(head_add_rel.size()[0], 1, head_add_rel.size()[1])  # [L, 1, H]
            head_add_rel = head_add_rel.repeat(1, node_emb.size()[0], 1)

            node_emb = node_emb.view(1, node_emb.size()[0], node_emb.size()[1])  # [1, N, H]
            node_emb = node_emb.repeat(head_add_rel.size()[0], 1, 1)

            result = head_add_rel - node_emb  # head+rel-tail [L, N, H]

        elif list_e_e_pair_idx is not None:
            ent_head_idxs = torch.LongTensor([x[0] for x in list_e_e_pair_idx])
            ent_tail_idxs = torch.LongTensor([x[1] for x in list_e_e_pair_idx])

            selected_ent_head_embs = node_emb[ent_head_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx
            selected_ent_tail_embs = rel_emb[ent_tail_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx

            selected_ent_head_embs = F.normalize(selected_ent_head_embs, 2, -1)
            selected_ent_tail_embs = F.normalize(selected_ent_tail_embs, 2, -1)
            rel_emb = F.normalize(rel_emb, 2, -1)

            head_sub_tail = selected_ent_head_embs - selected_ent_tail_embs  # [L, H]
            head_sub_tail = head_sub_tail.view(head_sub_tail.size()[0], 1, head_sub_tail.size()[1])  # [L, 1, H]
            head_sub_tail = head_sub_tail.repeat(1, rel_emb.size()[0], 1)  # [L, N, H]

            rel_emb = rel_emb.view(1, rel_emb.size()[0], rel_emb.size()[1])  # [1, N, H]
            rel_emb = rel_emb.repeat(head_sub_tail.size()[0], 1, 1)  # [L, N, H]

            result = head_sub_tail + rel_emb  # head-tail+rel [L, N, H]

        if self.loss_name in ['SoftMarginLoss']:
            # target labels are numbers selecting from -1 and 1.
            pred = torch.norm(result, self.p_norm, dim=2)  # TODO
        else:
            pred = torch.softmax(torch.norm(result, self.p_norm, dim=2), dim=-1)  # logits [L, N]

        if multi_label is not None:
            idxs_pos = torch.nonzero(multi_label == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(multi_label == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]
            return pred, pred_pos, pred_neg
        else:
            return pred