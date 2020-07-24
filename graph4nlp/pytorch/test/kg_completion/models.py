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
path_dir = os.getcwd()
random.seed(123)


# timer = CUDATimer()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel, X, A): # X and A haven't been used here.

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
        pred = F.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A): # X and A haven't been used here.
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred


class MarginLoss(nn.Module):

    def __init__(self, adv_temperature=None, margin=6.0):
        super(MarginLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(
                dim=-1).mean() + self.margin
        else:
            return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

# class TransE(nn.Module):
#
#     def __init__(self,
#                  num_entities=None,
#                  num_relations=None,
#                  p_norm=1,
#                  embedding_dim=None):
#         super(TransE, self).__init__()
#         self.p_norm = p_norm
#         self.ent_emb = nn.Embedding(num_entities, Config.embedding_dim)
#         self.rel_emb = nn.Embedding(num_relations, Config.embedding_dim)
#         # self.reset_parameters()
#         # self.loss = torch.nn.MultiLabelSoftMarginLoss()
#         self.loss = MarginLoss(margin = 5.0)
#         # self.loss = torch.nn.BCELoss()
#
#
#     def init(self):
#         nn.init.xavier_normal_(self.ent_emb.weight.data)
#         nn.init.xavier_normal_(self.rel_emb.weight.data)
#
#
#     def forward(self, e1, rel, e2_multi_1, e2_multi_0):
#         pos_triples = []
#         for h, r, t in zip(e1.squeeze().tolist(),
#                                      rel.squeeze().tolist(), e2_multi_1):
#             if isinstance(t, list):
#                 # for it in t:
#                 #     pos_triples.append((h,r,it))
#                 pos_triples.append((h,r,t[random.randint(0, len(t)-1)]))
#             else:
#                 pos_triples.append((h,r,t))
#
#         neg_triples = []
#         for h, r, t in zip(e1.squeeze().tolist(),
#                                      rel.squeeze().tolist(), e2_multi_0):
#             if isinstance(t, list):
#                 # for it in t:
#                 #     neg_triples.append((h,r,it))
#                 neg_triples.append((h, r, t[random.randint(0, len(t)-1)]))
#             else:
#                 neg_triples.append((h,r,t))
#
#         # positive
#         pos_h = torch.LongTensor([x[0] for x in pos_triples])
#         pos_r = torch.LongTensor([x[1] for x in pos_triples])
#         pos_t = torch.LongTensor([x[2] for x in pos_triples])
#
#         pos_h_embedded = self.ent_emb(pos_h)
#         pos_r_embedded = self.rel_emb(pos_r)
#         pos_t_embedded = self.ent_emb(pos_t)
#
#         pos_h_embedded = F.normalize(pos_h_embedded, 2, -1)
#         pos_r_embedded = F.normalize(pos_r_embedded, 2, -1)
#         pos_t_embedded = F.normalize(pos_t_embedded, 2, -1)
#
#         pos_score = pos_h_embedded + (pos_r_embedded - pos_t_embedded)
#         pos_score = torch.norm(pos_score, self.p_norm, dim=-1)
#
#         # positive
#         neg_h = torch.LongTensor([x[0] for x in neg_triples])
#         neg_r = torch.LongTensor([x[1] for x in neg_triples])
#         try:
#             neg_t = torch.LongTensor([x[2] for x in neg_triples])
#         except:
#             a = 0
#
#         neg_h_embedded = self.ent_emb(neg_h)
#         neg_r_embedded = self.rel_emb(neg_r)
#         neg_t_embedded = self.ent_emb(neg_t)
#
#         neg_h_embedded = F.normalize(neg_h_embedded, 2, -1)
#         neg_r_embedded = F.normalize(neg_r_embedded, 2, -1)
#         neg_t_embedded = F.normalize(neg_t_embedded, 2, -1)
#
#         neg_score = neg_h_embedded + (neg_r_embedded - neg_t_embedded)
#         neg_score = torch.norm(neg_score, self.p_norm, dim=-1)  # [L, N]
#
#
#         # head_add_rel = e1_embedded + rel_embedded  # [L, H]
#         # head_add_rel = head_add_rel.view(head_add_rel.size()[0], 1, head_add_rel.size()[1])  # [L, 1, H]
#         # head_add_rel = head_add_rel.repeat(1, node_emb.size()[0], 1)
#         # node_emb = node_emb.view(1, node_emb.size()[0], node_emb.size()[1])  # [1, N, H]
#         # node_emb = node_emb.repeat(head_add_rel.size()[0], 1, 1)
#
#         # result = head_add_rel - node_emb  # head+rel-tail [L, N, H]
#         # pos_score = torch.norm(pos_score, self.p_norm, dim=2)  # [L, N]
#         # logits = F.normalize(logits, 2, dim=1)
#         # logits = F.sigmoid(logits)
#
#         return pos_score, neg_score

class TransE(nn.Module):

    def __init__(self,
                 num_entities=None,
                 num_relations=None,
                 p_norm=1,
                 embedding_dim=None):
        super(TransE, self).__init__()
        self.p_norm = p_norm
        self.ent_emb = nn.Embedding(num_entities, Config.embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, Config.embedding_dim)
        # self.reset_parameters()
        # self.loss = torch.nn.MultiLabelMarginLoss()
        self.loss = torch.nn.BCELoss()


    def init(self):
        nn.init.xavier_normal_(self.ent_emb.weight.data)
        nn.init.xavier_normal_(self.rel_emb.weight.data)


    def forward(self, e1, rel, X, A):
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
        logits = torch.softmax(torch.norm(result, self.p_norm, dim=2),dim=-1)  # [L, N]

        return logits


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368,Config.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A): 
        e1_embedded= self.emb_e(e1).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred

# GCN
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations+1, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2],adj[2]]), requires_grad = True)
        A = A + A.transpose(0, 1)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(A, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# ConvTransE
class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvTransE, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.init_emb_size, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()

        self.conv1 =  nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding= int(math.floor(Config.kernel_size/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.init_emb_size)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.init_emb_size*Config.channels,Config.init_emb_size)
        #self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        #self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A):

        emb_initial = self.emb_e(X)
        e1_embedded_all = self.bn_init(emb_initial)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)

        return pred


# SACN
class SACN(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(SACN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.conv1 =  nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding= int(math.floor(Config.kernel_size/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim*Config.channels,Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A):

        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)

        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)

        return pred


from torch import nn
import torch
# from ..base import KGCompletionLayerBase

class KGCompletionLayerBase(nn.Module):

    def __init__(self):
        super(KGCompletionLayerBase, self).__init__()

    def forward(self, node_emb, rel_emb, list_e_r_pair_idx=None, list_e_e_pair_idx=None):
        raise NotImplementedError()


class DisMultLayer(KGCompletionLayerBase):
    r"""Specific class for knowledge graph completion task.
    DisMult from paper `Embedding entities and relations for learning and
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

    """

    def __init__(self,
                 input_dropout=0.0,
                 rel_emb_from_gnn=True,
                 num_relations=None,
                 embedding_dim=None):
        super(DisMultLayer, self).__init__()
        self.rel_emb_from_gnn = rel_emb_from_gnn
        self.inp_drop = nn.Dropout(input_dropout)
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
            N: number of nodes
            H: length of the node embeddings (entity embeddings)

        rel_emb: tensor [N_r,H]
            N: number of relations
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

        logits = torch.sigmoid(logits)

        return logits


class DistMultGNN(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMultGNN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        # self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        # self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        # self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        # self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        # self.conv1 =  nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding= int(math.floor(Config.kernel_size/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        # self.bn0 = torch.nn.BatchNorm1d(2)
        # self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        # self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim*Config.channels,Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        # self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)

        self.dismult_layer = DisMultLayer(rel_emb_from_gnn=False,
                                          num_relations=num_relations,
                                          embedding_dim=Config.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        # xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A):

        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)

        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
        # e1_embedded = e1_embedded_all[e1]
        # rel_embedded = self.emb_rel(rel)

        list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))

        pred = self.dismult_layer(e1_embedded_all, list_e_r_pair_idx = list_e_r_pair_idx)
        # pred = self.dismult_layer(e1_embedded_all, self.emb_rel.weight, list_e_r_pair_idx)

        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        # stacked_inputs = self.bn0(stacked_inputs)
        # x= self.inp_drop(stacked_inputs)
        # x= self.conv1(x)
        # x= self.bn1(x)
        # x= F.relu(x)
        # x = self.feature_map_drop(x)
        # x = x.view(Config.batch_size, -1)
        # x = self.fc(x)
        # x = self.hidden_drop(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        # pred = F.sigmoid(x)

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

            selected_ent_embs = F.normalize(selected_ent_embs, 2, -1)
            selected_rel_embs = F.normalize(selected_rel_embs, 2, -1)
            node_emb = F.normalize(node_emb, 2, -1)

            head_add_rel = selected_ent_embs + selected_rel_embs  # [L, H]
            head_add_rel = head_add_rel.view(head_add_rel.size()[0], 1, head_add_rel.size()[1])  # [L, 1, H]
            head_add_rel = head_add_rel.repeat(1, node_emb.size()[0], 1)

            node_emb = node_emb.view(1, node_emb.size()[0], node_emb.size()[1])  # [1, N, H]
            node_emb = node_emb.repeat(head_add_rel.size()[0], 1, 1)

            result = head_add_rel - node_emb  # head+rel-tail [L, N, H]
            # logits = torch.norm(result, self.p_norm, dim=2)  # [L, N]
            logits = torch.softmax(torch.norm(result, self.p_norm, dim=2), dim=-1)  # [L, N]

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
            # logits = torch.norm(result, self.p_norm, dim=2)  # [L, N]
            logits = torch.softmax(torch.norm(result, self.p_norm, dim=2), dim=-1)  # [L, N]

        # logits = torch.sigmoid(logits)

        return logits


class TransEGNN(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(TransEGNN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        # self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        # self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        # self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        # self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        # self.conv1 =  nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding= int(math.floor(Config.kernel_size/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        # self.bn0 = torch.nn.BatchNorm1d(2)
        # self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        # self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim*Config.channels,Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        # self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)

        self.transe_layer = TransELayer(rel_emb_from_gnn=False,
                                          num_relations=num_relations,
                                          embedding_dim=Config.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        # xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A):

        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)

        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
        # e1_embedded = e1_embedded_all[e1]
        # rel_embedded = self.emb_rel(rel)

        list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))

        pred = self.transe_layer(e1_embedded_all, list_e_r_pair_idx = list_e_r_pair_idx)
        # pred = self.dismult_layer(e1_embedded_all, self.emb_rel.weight, list_e_r_pair_idx)

        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        # stacked_inputs = self.bn0(stacked_inputs)
        # x= self.inp_drop(stacked_inputs)
        # x= self.conv1(x)
        # x= self.bn1(x)
        # x= F.relu(x)
        # x = self.feature_map_drop(x)
        # x = x.view(Config.batch_size, -1)
        # x = self.fc(x)
        # x = self.hidden_drop(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        # pred = F.sigmoid(x)

        return pred