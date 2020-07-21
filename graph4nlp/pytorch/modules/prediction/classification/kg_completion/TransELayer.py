from torch import nn
import torch
from ..base import KGCompletionLayerBase


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

    """

    def __init__(self,
                 p_norm=1,
                 rel_emb_from_gnn=True,
                 num_relations=None,
                 embedding_dim=None):
        super(TransELayer, self).__init__()
        self.p_norm = p_norm
        self.rel_emb_from_gnn = rel_emb_from_gnn
        if self.rel_emb_from_gnn == False:
            assert num_relations != None
            assert embedding_dim != None
            self.rel_emb = nn.Embedding(num_relations, embedding_dim)
            self.reset_parameters()


    def reset_parameters(self):
        if self.rel_emb_from_gnn == False:
            nn.init.xavier_normal_(self.rel_emb.weight.data)


    def forward(self, node_emb, rel_emb=None, list_e_r_pair_idx=None, list_e_e_pair_idx=None):
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

        Returns
        -------
             logit tensor: [N, num_class] The score logits for all nodes preidcted.
        """
        if self.rel_emb_from_gnn == False:
            assert rel_emb == None
            rel_emb = self.rel_emb.weight

        if list_e_r_pair_idx == None and list_e_e_pair_idx == None:
            raise RuntimeError("Only one of `list_e_r_pair_idx` and `list_e_e_pair_idx` can be `None`.")

        assert node_emb.size()[1]==rel_emb.size()[1]

        if list_e_r_pair_idx != None:
            ent_idxs = torch.LongTensor([x[0] for x in list_e_r_pair_idx])
            rel_idxs = torch.LongTensor([x[1] for x in list_e_r_pair_idx])

            selected_ent_embs = node_emb[ent_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx
            selected_rel_embs = rel_emb[rel_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx

            head_add_rel = selected_ent_embs + selected_rel_embs  # [L, H]
            head_add_rel = head_add_rel.view(head_add_rel.size()[0], 1, head_add_rel.size()[1])  # [L, 1, H]
            node_emb = node_emb.view(1, node_emb.size()[0], node_emb.size()[1])  # [1, N, H]

            result = head_add_rel - node_emb  # head+rel-tail [L, N, H]
            logits = torch.norm(result, self.p_norm, dim=2)  # [L, N]

        elif list_e_e_pair_idx != None:
            ent_head_idxs = torch.LongTensor([x[0] for x in list_e_e_pair_idx])
            ent_tail_idxs = torch.LongTensor([x[1] for x in list_e_e_pair_idx])

            selected_ent_head_embs = node_emb[ent_head_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx
            selected_ent_tail_embs = rel_emb[ent_tail_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx

            head_sub_tail = selected_ent_head_embs - selected_ent_tail_embs  # [L, H]
            head_sub_tail = head_sub_tail.view(head_sub_tail.size()[0], 1, head_sub_tail.size()[1])  # [L, 1, H]
            rel_emb = rel_emb.view(1, rel_emb.size()[0], rel_emb.size()[1])  # [1, N, H]

            result = head_sub_tail + rel_emb  # head-tail+rel [L, N, H]
            logits = torch.norm(result, self.p_norm, dim=2)  # [L, N]

        logits = torch.sigmoid(logits)

        return logits
