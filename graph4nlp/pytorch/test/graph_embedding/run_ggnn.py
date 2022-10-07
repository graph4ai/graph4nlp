import argparse
import os
import time
import dgl
import networkx as nx
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data, register_data_args

from ...data.data import GraphData
from ...modules.graph_embedding_learning.ggnn import GGNN
from .utils import EarlyStopping


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, g, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


class GNNClassifier(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        output_size,
        n_etypes,
        n_class,
        direction_option,
        drop=0.0,
        use_edge_weight=False,
    ):
        super(GNNClassifier, self).__init__()
        self.direction_option = direction_option
        self.model = GGNN(
            num_layers,
            input_size,
            output_size,
            n_etypes=n_etypes,
            dropout=drop,
            direction_option=direction_option,
            use_edge_weight=use_edge_weight,
        )

        if self.direction_option == "bi_sep":
            self.fc = nn.Linear(2 * output_size, n_class)
        else:
            self.fc = nn.Linear(output_size, n_class)

    def forward(self, graph):
        graph = self.model(graph)
        logits = graph.node_features["node_emb"]
        logits = self.fc(F.elu(logits))

        return logits


def prepare_dgl_graph_data(args):
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

    num_feats = features.shape[1]
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

    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()

    data = {
        "features": features,
        "graph": g,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "labels": labels,
        "num_feats": num_feats,
        "n_classes": n_classes,
        "n_edges": n_edges,
    }

    return data


def prepare_ogbn_graph_data(args):
    from ogb.nodeproppred import DglNodePropPredDataset

    dataset = DglNodePropPredDataset(name=args.dataset)

    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        torch.LongTensor(split_idx["train"]),
        torch.LongTensor(split_idx["valid"]),
        torch.LongTensor(split_idx["test"]),
    )
    g, labels = dataset[
        0
    ]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    features = torch.Tensor(g.ndata["feat"])
    labels = torch.LongTensor(labels).squeeze(-1)

    # add self loop
    # no duplicate self loop will be added for nodes already having self loops
    new_g = dgl.transform.add_self_loop(g)

    # edge_index = data[0]['edge_index']
    # adj = to_undirected(edge_index, num_nodes=data[0]['num_nodes'])
    # assert adj.diagonal().sum() == 0 and adj.max() <= 1 and (adj != adj.transpose()).sum() == 0

    num_feats = features.shape[1]
    n_classes = labels.max().item() + 1
    n_edges = new_g.number_of_edges()
    print(
        """----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d"""
        % (n_edges, n_classes, train_idx.shape[0], val_idx.shape[0], test_idx.shape[0])
    )

    data = {
        "features": features,
        "graph": new_g,
        "train_mask": train_idx,
        "val_mask": val_idx,
        "test_mask": test_idx,
        "labels": labels,
        "num_feats": num_feats,
        "n_classes": n_classes,
        "n_edges": n_edges,
    }

    return data


def main(args, seed):
    # load and preprocess dataset
    if args.dataset.startswith("ogbn"):
        # Open Graph Benchmark datasets
        data = prepare_ogbn_graph_data(args)
    else:
        # DGL datasets
        data = prepare_dgl_graph_data(args)

    features, dgl_graph, train_mask, val_mask, test_mask, labels, num_feats, n_classes, n_edges = (
        data["features"],
        data["graph"],
        data["train_mask"],
        data["val_mask"],
        data["test_mask"],
        data["labels"],
        data["num_feats"],
        data["n_classes"],
        data["n_edges"],
    )

    # Configure
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not args.no_cuda and torch.cuda.is_available():
        print("[ Using CUDA ]")
        device = torch.device("cuda" if args.gpu < 0 else "cuda:%d" % args.gpu)
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    dgl_graph.ndata["node_feat"] = features

    # convert DGLGraph to GraphData
    g = GraphData()
    g.from_dgl(dgl_graph)

    edge_types = []
    directed_edges = []
    for edge in g.edges():
        if edge not in directed_edges and (edge[1], edge[0]) not in directed_edges:
            directed_edges.append(edge)
            edge_types.append(0)
        else:
            edge_types.append(1)

    if args.num_etypes == 2:
        g.edge_features["etype"] = torch.tensor(edge_types, dtype=torch.long).to(device)
    # else:
    #     g.edge_features['etype'] = torch.LongTensor([0] * g.get_edge_num()).to(device)
    g.edge_features["edge_weight"] = (
        torch.tensor([1] * g.get_edge_num(), dtype=torch.float32).view(-1, 1).to(device)
    )
    g.edge_features["reverse_edge_weight"] = (
        torch.tensor([1] * g.get_edge_num(), dtype=torch.float32).view(-1, 1).to(device)
    )

    # create model
    model = GNNClassifier(
        args.num_layers,
        num_feats,
        args.num_hidden,
        args.num_etypes,
        n_classes,
        args.direction_option,
        drop=args.drop,
        use_edge_weight=args.use_edge_weight,
    )

    print(model)
    model.to(device)

    if args.early_stop:
        stopper = EarlyStopping("{}.{}".format(args.save_model_path, seed), patience=args.patience)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, g, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
            " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(), train_acc, val_acc, n_edges / np.mean(dur) / 1000
            )
        )

    print()
    if args.early_stop:
        model = stopper.load_checkpoint(model)
        print("Restored best saved model")
        os.remove(stopper.save_model_path)
        print("Removed best saved model file to save disk space")

    acc = evaluate(model, g, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GGNN")
    register_data_args(parser)
    parser.add_argument("--num-runs", type=int, default=5, help="number of runs")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="use CPU")
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument(
        "--direction-option",
        type=str,
        default="undirected",
        help="direction type (`undirected`, `bi_fuse`, `bi_sep`)",
    )
    parser.add_argument("--num-heads", type=int, default=8, help="number of hidden attention heads")
    parser.add_argument(
        "--num-out-heads", type=int, default=1, help="number of output attention heads"
    )
    parser.add_argument("--num-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num-etypes", type=int, default=2, help="number of edge types")
    parser.add_argument("--num-hidden", type=int, default=1435, help="number of hidden units")
    parser.add_argument(
        "--residual", action="store_true", default=False, help="use residual connection"
    )
    parser.add_argument("--drop", type=float, default=0.0, help="input feature dropout")
    parser.add_argument("--use_edge_weight", type=bool, default=False, help="use edge weight")
    # parser.add_argument("--attn-drop", type=float, default=.6,
    #                     help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--negative-slope", type=float, default=0.2, help="the negative slope of leaky relu"
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="indicates whether to use early stop or not",
    )
    parser.add_argument("--patience", type=int, default=100, help="early stopping patience")
    parser.add_argument(
        "--fastmode", action="store_true", default=False, help="skip re-evaluate the validation set"
    )
    parser.add_argument(
        "--save-model-path", type=str, default="checkpoint", help="path to the best saved model"
    )
    args = parser.parse_args()
    args.save_model_path = "{}_{}_{}_{}".format(
        args.save_model_path, args.dataset, "ggnn", args.direction_option
    )
    print(args)

    np.random.seed(123)
    scores = []
    for _ in range(args.num_runs):
        seed = np.random.randint(10000)
        scores.append(main(args, seed))

    print(
        "\nTest Accuracy ({} runs): mean {:.4f}, std {:.4f}".format(
            args.num_runs, np.mean(scores), np.std(scores)
        )
    )
