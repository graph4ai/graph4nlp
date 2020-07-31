from torch import nn
import torch
from ..base import KGCompletionLayerBase


class ComplExLayer(KGCompletionLayerBase):
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
        super(ComplExLayer, self).__init__()
        self.rel_emb_from_gnn = rel_emb_from_gnn
        self.inp_drop = nn.Dropout(input_dropout)
        if self.rel_emb_from_gnn == False:
            assert num_relations != None
            assert embedding_dim != None
            self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim)
            self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim)
            self.reset_parameters()
        self.loss_name = loss_name
        self.reset_parameters()


    def reset_parameters(self):
        if self.rel_emb_from_gnn == False:
            nn.init.xavier_normal_(self.emb_rel_real.weight.data)
            nn.init.xavier_normal_(self.emb_rel_img.weight.data)


    def forward(self,
                node_emb_real,
                node_emb_img,
                rel_emb_real=None,
                rel_emb_img=None,
                list_e_r_pair_idx=None,
                list_e_e_pair_idx=None,
                multi_label=None):
        if self.rel_emb_from_gnn == False:
            assert rel_emb_real == None
            assert rel_emb_img == None
            rel_emb_real = self.emb_rel_real.weight
            rel_emb_img = self.emb_rel_img.weight

        if list_e_r_pair_idx == None and list_e_e_pair_idx == None:
            raise RuntimeError("Only one of `list_e_r_pair_idx` and `list_e_e_pair_idx` can be `None`.")

        assert node_emb_real.size()[1]==rel_emb_real.size()[1]
        assert rel_emb_img.size()[1]==rel_emb_img.size()[1]

        if list_e_r_pair_idx != None:
            ent_idxs = torch.LongTensor([x[0] for x in list_e_r_pair_idx])
            rel_idxs = torch.LongTensor([x[1] for x in list_e_r_pair_idx])

            selected_ent_embs_real = node_emb_real[ent_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx
            selected_ent_embs_img = node_emb_img[ent_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx
            selected_rel_embs_real = rel_emb_real[rel_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx
            selected_rel_embs_img = rel_emb_img[rel_idxs].squeeze()  # [L, H]. L is the length of list_e_r_pair_idx

            # dropout
            selected_ent_embs_real = self.inp_drop(selected_ent_embs_real)
            selected_ent_embs_img = self.inp_drop(selected_ent_embs_img)
            selected_rel_embs_real = self.inp_drop(selected_rel_embs_real)
            selected_rel_embs_img = self.inp_drop(selected_rel_embs_img)

            # complex space bilinear product (equivalent to HolE)
            realrealreal = torch.mm(selected_ent_embs_real * selected_rel_embs_real,
                                    node_emb_real.transpose(1, 0))
            realimgimg = torch.mm(selected_ent_embs_real * selected_rel_embs_img,
                                  node_emb_img.transpose(1, 0))
            imgrealimg = torch.mm(selected_ent_embs_img * selected_rel_embs_real,
                                  node_emb_img.transpose(1, 0))
            imgimgreal = torch.mm(selected_ent_embs_img * selected_rel_embs_img,
                                  node_emb_real.transpose(1, 0))
            pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        elif list_e_e_pair_idx != None:
            # ent_head_idxs = torch.LongTensor([x[0] for x in list_e_e_pair_idx])
            # ent_tail_idxs = torch.LongTensor([x[1] for x in list_e_e_pair_idx])
            #
            # selected_ent_head_embs = node_emb[ent_head_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx
            # selected_ent_tail_embs = rel_emb[ent_tail_idxs].squeeze()  # [L, H]. L is the length of list_e_e_pair_idx
            #
            # # dropout
            # selected_ent_head_embs = self.inp_drop(selected_ent_head_embs)
            # selected_ent_tail_embs = self.inp_drop(selected_ent_tail_embs)
            #
            # logits = torch.mm(selected_ent_head_embs*selected_ent_tail_embs,
            #                   rel_emb.transpose(1, 0))
            raise NotImplementedError()

        if self.loss_name in ["SoftMarginLoss"]:
            pred = torch.tanh(pred)
        else:
            pred = torch.sigmoid(pred)

        if type(multi_label) != type(None):
            idxs_pos = torch.nonzero(multi_label == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(multi_label == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]

            return pred, pred_pos, pred_neg
        else:
            return pred