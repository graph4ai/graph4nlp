import torch
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.init import xavier_normal_

from graph4nlp.pytorch.modules.graph_embedding_learning.gcn import GCN
from graph4nlp.pytorch.modules.graph_embedding_learning.ggnn import GGNN
from graph4nlp.pytorch.modules.prediction.classification.kg_completion import ComplEx, DistMult


class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel, kg_graph=None):

        e1_embedded_real = self.emb_e_real(e1).squeeze(1)
        rel_embedded_real = self.emb_rel_real(rel).squeeze(1)
        e1_embedded_img = self.emb_e_img(e1).squeeze(1)
        rel_embedded_img = self.emb_rel_img(rel).squeeze(1)

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(
            e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0)
        )
        realimgimg = torch.mm(
            e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0)
        )
        imgrealimg = torch.mm(
            e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0)
        )
        imgimgreal = torch.mm(
            e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0)
        )
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class Distmult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Distmult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, kg_graph=None):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze(1)
        rel_embedded = rel_embedded.squeeze(1)

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter("b", Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size, args.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, kg_graph=None):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class GGNNDistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_layers=2):
        super(GGNNDistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.num_layers = num_layers
        self.gnn = GGNN(
            self.num_layers,
            args.embedding_dim,
            args.embedding_dim,
            args.embedding_dim,
            feat_drop=args.input_drop,
            direction_option=args.direction_option,
        )
        self.direction_option = args.direction_option

        self.loss = torch.nn.BCELoss()
        self.distmult = DistMult(args.input_drop, loss_name="BCELoss")

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, kg_graph):
        X = torch.LongTensor(list(range(self.num_entities))).to(e1.device)

        kg_graph.node_features["node_feat"] = self.emb_e(X)
        kg_graph = self.gnn(kg_graph)

        e1_embedded = kg_graph.node_features["node_feat"][e1]
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze(1)
        rel_embedded = rel_embedded.squeeze(1)

        kg_graph = self.distmult(kg_graph, e1_embedded, rel_embedded, self.emb_e)
        logits = kg_graph.graph_attributes["logits"]

        return logits


class GCNDistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_layers=2):
        super(GCNDistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)

        self.num_entities = num_entities
        self.num_relations = num_relations

        self.num_layers = num_layers
        self.gnn = GCN(
            self.num_layers,
            args.embedding_dim,
            args.embedding_dim,
            args.embedding_dim,
            args.direction_option,
            feat_drop=args.input_drop,
        )

        self.direction_option = args.direction_option
        self.distmult = DistMult(args.input_drop, loss_name="BCELoss")
        self.loss = torch.nn.BCELoss()
        # self.loss = KGLoss('SigmoidLoss')

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, kg_graph=None):
        X = torch.LongTensor(list(range(self.num_entities))).to(e1.device)

        kg_graph.node_features["node_feat"] = self.emb_e(X)
        kg_graph = self.gnn(kg_graph)

        e1_embedded = kg_graph.node_features["node_feat"][e1]
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze(1)
        rel_embedded = rel_embedded.squeeze(1)

        kg_graph = self.distmult(kg_graph, e1_embedded, rel_embedded, self.emb_e)
        logits = kg_graph.graph_attributes["logits"]

        return logits


class GCNComplex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_layers=2):
        super(GCNComplex, self).__init__()
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)

        self.num_entities = num_entities
        self.num_relations = num_relations

        self.num_layers = num_layers
        self.gnn = GCN(
            self.num_layers,
            args.embedding_dim,
            args.embedding_dim,
            args.embedding_dim,
            args.direction_option,
            feat_drop=args.input_drop,
        )

        self.direction_option = args.direction_option
        self.loss = torch.nn.BCELoss()
        self.complex = ComplEx(args.input_drop)

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel, kg_graph=None):
        X = torch.LongTensor(list(range(self.num_entities))).to(e1.device)

        kg_graph.node_features["node_feat"] = self.emb_e_real(X)
        kg_graph = self.gnn(kg_graph)
        e1_embedded_real = kg_graph.node_features["node_feat"][e1].squeeze(1)

        kg_graph.node_features["node_feat"] = self.emb_e_img(X)
        kg_graph = self.gnn(kg_graph)
        e1_embedded_img = kg_graph.node_features["node_feat"][e1].squeeze(1)

        rel_embedded_real = self.emb_rel_real(rel).squeeze(1)
        rel_embedded_img = self.emb_rel_img(rel).squeeze(1)

        kg_graph = self.complex(
            kg_graph,
            e1_embedded_real,
            rel_embedded_real,
            e1_embedded_img,
            rel_embedded_img,
            self.emb_e_real,
            self.emb_e_img,
        )
        logits = kg_graph.graph_attributes["logits"]

        return logits
