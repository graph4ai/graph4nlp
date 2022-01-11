import argparse
import time
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.data import load_data, register_data_args

from ...data.data import GraphData
from ...modules.graph_embedding_learning.graphsage import GraphSAGE
from ...modules.prediction.classification.node_classification.BiLSTMFeedForwardNN import (
    BiLSTMFeedForwardNN,
)


class GCN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, aggreagte_type, output_size, dropout
    ):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.model = GraphSAGE(
            n_layers,
            in_feats,
            n_hidden,
            output_size,
            aggreagte_type,
            direction_option=args.direction_option,
            feat_drop=0.6,
            bias=True,
            activation=nn.ReLU,
        )
        # output layer
        # self.classifier=FeedForwardNN(output_size,n_classes,[16],nn.Sigmoid())
        self.classifier = BiLSTMFeedForwardNN(output_size, n_classes, 16)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph):
        out_graph = self.model(graph)
        out_graph_ = self.classifier(out_graph)
        return out_graph_.node_features["logits"]


def evaluate(model, G, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(G)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, "BoolTensor"):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print(
        """----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d"""
        % (
            n_edges,
            n_classes,
            train_mask.int().sum().item(),
            val_mask.int().sum().item(),
            test_mask.int().sum().item(),
        )
    )

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata["node_feat"] = features
    if cuda:
        norm = norm.cuda()
    g.ndata["norm"] = norm.unsqueeze(1)

    G = GraphData()
    G.from_dgl(g)

    # create GCN model
    model = GCN(
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        args.aggregate_type,
        args.output_size,
        dropout=args.dropout,
    )

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(G)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, G, labels, val_mask)
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000
            )
        )

    print()
    acc = evaluate(model, G, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=400, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=[64], help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action="store_true", help="graph self-loop (default=False)")
    parser.add_argument(
        "--aggregate_type",
        type=str,
        default="mean",
        help="aggregate type: 'mean','gcn','pool','lstm'",
    )
    parser.add_argument("--output_size", type=int, default=16, help="hiddensize")
    parser.add_argument(
        "--direction_option",
        type=str,
        default="uni",
        help="direction type (`uni`, `bi_fuse`, `bi_sep`)",
    )
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    main(args)
