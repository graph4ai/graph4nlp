import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution

from ...modules.prediction.classification.link_prediction.ConcatFeedForwardNNLayer import (
    ConcatFeedForwardNNLayer,
)
from ...modules.prediction.classification.link_prediction.ElementSumLayer import ElementSumLayer
from ...modules.prediction.classification.link_prediction.StackedElementProdLayer import (
    StackedElementProdLayer,
)


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, prediction_type):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.prediction_type = prediction_type
        self.act = lambda x: x
        if self.prediction_type == "ele_sum":
            self.dc = ElementSumLayer(hidden_dim2, 16, 1)
        if self.prediction_type == "concat_NN":
            self.dc = ConcatFeedForwardNNLayer(hidden_dim2, 16, 1)
        if self.prediction_type == "stacked_ele_prod":
            self.dc = StackedElementProdLayer(hidden_dim2, 16, 1, 1)
        # InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        if self.prediction_type == "stacked_ele_prod":
            link_logits = self.dc([z])
            recovered = self.dc([mu])
        else:
            link_logits = self.dc(z)
            recovered = self.dc(mu)
        return self.act(link_logits), mu, logvar, recovered


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
