import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from src.spodernet.spodernet.utils.global_config import Config
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from torch.nn.init import xavier_normal_, xavier_uniform_
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.init as init
import os, sys
import random
import numpy as np
from models import GraphConvolution
from models import MarginLoss
path_dir = os.getcwd()
random.seed(123)


# timer = CUDATimer()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations, loss_name='BCELoss'):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)

        self.loss_name = loss_name
        if loss_name == 'BCELoss':
            self.loss = torch.nn.BCELoss()
        elif loss_name == "SoftplusLoss":
            self.loss = SoftplusLoss()
        elif loss_name == "SigmoidLoss":
            self.loss = SigmoidLoss()
        elif loss_name == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        elif loss_name == "MSELoss":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

        self.init()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel, X, A, e2_multi=None): # X and A haven't been used here.

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal

        if self.loss_name in ["SoftMarginLoss"]:
            pred = torch.tanh(pred)
        else:
            pred = torch.sigmoid(pred)

        # if e2_multi!=None:
        if type(e2_multi) != type(None):
            idxs_pos = torch.nonzero(e2_multi == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(e2_multi == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]

            return pred, pred_pos, pred_neg
        else:
            return pred


class KGCompletionLayerBase(nn.Module):

    def __init__(self):
        super(KGCompletionLayerBase, self).__init__()

    def forward(self, node_emb, rel_emb, list_e_r_pair_idx=None, list_e_e_pair_idx=None):
        raise NotImplementedError()


class ComplexLayer(KGCompletionLayerBase):
    def __init__(self,
                 input_dropout=0.0,
                 rel_emb_from_gnn=True,
                 num_relations=None,
                 embedding_dim=None,
                 loss_name='BCELoss'):
        super(ComplexLayer, self).__init__()
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
            raise NotImplementedError

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


class ComplexGNN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, loss_name='BCELoss'):
        super(ComplexGNN, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim)

        self.gc1 = GraphConvolution(Config.embedding_dim, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        self.bn1 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)

        self.gc3 = GraphConvolution(Config.embedding_dim, Config.gc1_emb_size, num_relations)
        self.gc4 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)

        # self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim)
        # self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim)
        # self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.complex_layer = ComplexLayer(rel_emb_from_gnn=False,
                                          num_relations=num_relations,
                                          embedding_dim=Config.embedding_dim,
                                          loss_name=loss_name)

        self.loss_name = loss_name
        if loss_name == 'BCELoss':
            self.loss = torch.nn.BCELoss()
        elif loss_name == "SoftplusLoss":
            self.loss = SoftplusLoss()
        elif loss_name == "SigmoidLoss":
            self.loss = SigmoidLoss()
        elif loss_name == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        elif loss_name == "MSELoss":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

        self.init()

    # def reset_parameters(self):
    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        # xavier_normal_(self.emb_rel_real.weight.data)
        # xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel, X, A, e2_multi=None): # X and A haven't been used here.

        e1_embedded_real = self.emb_e_real(X)
        e1_embedded_img = self.emb_e_img(X)

        x_real = self.gc1(e1_embedded_real, A)
        x_real = self.bn1(x_real)
        x_real = torch.tanh(x_real)
        x_real = torch.dropout(x_real, Config.dropout_rate, train=self.training)
        x_real = self.bn2(self.gc2(x_real, A))
        x_real = torch.tanh(x_real)
        e1_embedded_all_real = torch.dropout(x_real, Config.dropout_rate, train=self.training)

        x_img = self.gc3(e1_embedded_img, A)
        x_img = self.bn3(x_img)
        x_img = torch.tanh(x_img)
        x_img = torch.dropout(x_img, Config.dropout_rate, train=self.training)
        x_img = self.bn4(self.gc4(x_img, A))
        x_img = torch.tanh(x_img)
        e1_embedded_all_img = torch.dropout(x_img, Config.dropout_rate, train=self.training)

        list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))
        pred = self.complex_layer(e1_embedded_all_real,
                                  e1_embedded_all_img,
                                  list_e_r_pair_idx = list_e_r_pair_idx,
                                  multi_label=e2_multi)

        return pred


class TransE(nn.Module):
    def __init__(self,
                 num_entities=None,
                 num_relations=None,
                 p_norm=1,
                 loss_name='BCELoss'
                 ):
        super(TransE, self).__init__()
        self.p_norm = p_norm
        self.ent_emb = nn.Embedding(num_entities, Config.embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, Config.embedding_dim)

        self.loss_name = loss_name
        if loss_name == 'BCELoss':
            self.loss = torch.nn.BCELoss()
        elif loss_name == "SoftplusLoss":
            self.loss = SoftplusLoss()
        elif loss_name == "SigmoidLoss":
            self.loss = SigmoidLoss()
        elif loss_name == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        elif loss_name == "MSELoss":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

        self.init()


    def init(self):
        nn.init.xavier_normal_(self.ent_emb.weight.data)
        nn.init.xavier_normal_(self.rel_emb.weight.data)


    def forward(self, e1, rel, X, A, e2_multi=None):
        e1_embedded = self.ent_emb(e1)
        rel_embedded = self.rel_emb(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()
        node_emb = self.ent_emb.weight

        e1_embedded = F.normalize(e1_embedded, 2, -1)
        rel_embedded = F.normalize(rel_embedded, 2, -1)
        node_emb = F.normalize(node_emb, 2, -1)

        head_add_rel = e1_embedded + rel_embedded  # [L, H]
        head_add_rel = head_add_rel.view(head_add_rel.size()[0], 1, head_add_rel.size()[1])  # [L, 1, H]
        head_add_rel = head_add_rel.repeat(1, node_emb.size()[0], 1)
        node_emb = node_emb.view(1, node_emb.size()[0], node_emb.size()[1])  # [1, N, H]
        node_emb = node_emb.repeat(head_add_rel.size()[0], 1, 1)

        result = head_add_rel - node_emb  # head+rel-tail [L, N, H]
        # logits = torch.softmax(torch.norm(result, self.p_norm, dim=2),dim=-1)  # [L, N]

        if self.loss_name in ["SoftMarginLoss"]:
            pred = torch.norm(result, self.p_norm, dim=2)
        else:
            pred = torch.softmax(torch.norm(result, self.p_norm, dim=2),dim=-1)  # [L, N]

        if e2_multi != None:
            idxs_pos = torch.nonzero(e2_multi == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(e2_multi == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]

            # return pred, pred_pos, pred_neg
            pred_pos = pred_pos.repeat(pred_neg.size()[0] // pred_pos.size()[0])
            return pred, pred_pos, pred_neg[:pred_pos.size()[0]]
        else:
            return pred

class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations, loss_name='BCELoss'):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

        self.loss_name = loss_name
        if loss_name == 'BCELoss':
            self.loss = torch.nn.BCELoss()
        elif loss_name == "SoftplusLoss":
            self.loss = SoftplusLoss()
        elif loss_name == "SigmoidLoss":
            self.loss = SigmoidLoss()
        elif loss_name == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        elif loss_name == "MSELoss":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

        self.init()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A, e2_multi=None): # X and A haven't been used here.
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        if self.loss_name in ["SoftMarginLoss"]:
            pred = torch.tanh(pred)
        else:
            pred = torch.sigmoid(pred)

        # if e2_multi!=None:
        if type(e2_multi) != type(None):
            idxs_pos = torch.nonzero(e2_multi == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(e2_multi == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]

            return pred, pred_pos, pred_neg
        else:
            return pred

class SoftplusLoss(nn.Module):

    def __init__(self, adv_temperature=None):
        super(SoftplusLoss, self).__init__()
        self.criterion = nn.Softplus()
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.criterion(-p_score).mean() + (self.get_weights(n_score) * self.criterion(n_score)).sum(
                dim=-1).mean()) / 2
        else:
            return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature = None):
        super(SigmoidLoss, self).__init__()
        self.criterion = nn.LogSigmoid()
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2
        else:
            return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

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

            # dropout
            selected_ent_embs = self.inp_drop(selected_ent_embs)
            selected_rel_embs = self.inp_drop(selected_rel_embs)

            logits = torch.mm(selected_ent_embs * selected_rel_embs,
                              node_emb.transpose(1, 0))
        elif list_e_e_pair_idx != None:
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

        # if multi_label!=None:
        if type(multi_label) != type(None):
            idxs_pos = torch.nonzero(multi_label == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(multi_label == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]
            return pred, pred_pos, pred_neg
        else:
            return pred

class DistMultGNN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, loss_name='BCELoss'):
        super(DistMultGNN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        # self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim)

        self.loss_name = loss_name
        if loss_name == 'BCELoss':  # Multi-Class Loss (Binary Cross Entropy Loss)
            self.loss = torch.nn.BCELoss()
        elif loss_name == "SoftplusLoss":
            self.loss = SoftplusLoss()
        elif loss_name == "SigmoidLoss":
            self.loss = SigmoidLoss()
        elif loss_name == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        elif loss_name == "MSELoss":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()


        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim*Config.channels,Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.dismult_layer = DistMultLayer(rel_emb_from_gnn=False,
                                            num_relations=num_relations,
                                            embedding_dim=Config.embedding_dim,
                                            loss_name=self.loss_name)
        print(num_entities, num_relations)
        self.init()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        # xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A, e2_multi=None):

        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = torch.dropout(x, Config.dropout_rate, train=self.training)

        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = torch.tanh(x)
        e1_embedded_all = torch.dropout(e1_embedded_all, Config.dropout_rate, train=self.training)
        # e1_embedded = e1_embedded_all[e1]
        # rel_embedded = self.emb_rel(rel)

        list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))


        # TODO: emb_rel from gnn
        pred = self.dismult_layer(e1_embedded_all, list_e_r_pair_idx = list_e_r_pair_idx, multi_label=e2_multi)
        # pred = self.dismult_layer(e1_embedded_all, self.emb_rel.weight, list_e_r_pair_idx, multi_label=e2_multi)

        return pred


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
             logit tensor: [N, num_class] The score logits for all nodes preidcted.
        """
        if self.rel_emb_from_gnn == False:
            assert rel_emb == None
            rel_emb = self.rel_emb.weight

        if list_e_r_pair_idx == None and list_e_e_pair_idx == None:
            raise RuntimeError("Only one of `list_e_r_pair_idx` and `list_e_e_pair_idx` can be `None`.")

        assert node_emb.size()[1] == rel_emb.size()[1]

        if list_e_r_pair_idx != None:
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

        elif list_e_e_pair_idx != None:
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

        if self.loss_name in ['SoftMarginLoss', 'MarginLoss']:
            # target labels are numbers selecting from -1 and 1.
            pred = torch.norm(result, self.p_norm, dim=2)  # TODO
        else:
            pred = torch.softmax(torch.norm(result, self.p_norm, dim=2), dim=-1)  # logits [L, N]

        if multi_label!=None:
            idxs_pos = torch.nonzero(multi_label == 1.)
            pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

            idxs_neg = torch.nonzero(multi_label == 0.)
            pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]
            return pred, pred_pos, pred_neg
        else:
            return pred


class TransEGNN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, loss_name = 'BCELoss'):
        super(TransEGNN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        # self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim)
        # self.loss = torch.nn.BCELoss()
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim*Config.channels,Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)

        # self.loss_name = "SoftplusLoss"  # similar to Pairwise Hinge Loss
        # self.loss = SoftplusLoss()

        # self.loss_name = "SigmoidLoss"  # -> Pointwise Logistic Loss
        # self.loss = SigmoidLoss()

        # self.loss_name = "MSELoss"  # -> Pointwise Square Error Loss
        # self.loss = torch.nn.MSELoss()

        # self.loss_name = "BCELoss"  # -> Multi-Class Loss (Binary Cross Entropy Loss)
        # self.loss = torch.nn.BCELoss()

        # self.loss_name = "MarginLoss"  # TODO
        # self.loss = MarginLoss()

        self.loss_name = loss_name
        if loss_name == 'BCELoss':
            # self.loss_name = "BCELoss"  # -> Multi-Class Loss (Binary Cross Entropy Loss)
            self.loss = torch.nn.BCELoss()
        elif loss_name == "SoftplusLoss":
            self.loss = SoftplusLoss()
        elif loss_name == "SigmoidLoss":
            self.loss = SigmoidLoss()
        elif loss_name == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        elif loss_name == "MSELoss":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

        # self.transe_layer = TransELayer(rel_emb_from_gnn=False,
        self.transe_layer = TransELayer(rel_emb_from_gnn=False,
                                        num_relations=num_relations,
                                        embedding_dim=Config.embedding_dim,
                                        loss_name=self.loss_name)
        print(num_entities, num_relations)
        self.init()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        # xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A, e2_multi=None):

        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = torch.dropout(x, Config.dropout_rate, train=self.training)

        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = torch.tanh(x)
        e1_embedded_all = torch.dropout(e1_embedded_all, Config.dropout_rate, train=self.training)

        list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))

        pred = self.transe_layer(e1_embedded_all, list_e_r_pair_idx = list_e_r_pair_idx, multi_label=e2_multi)
        # pred = self.transe_layer(e1_embedded_all, self.emb_rel.weight, list_e_r_pair_idx, multi_label=e2_multi)

        return pred