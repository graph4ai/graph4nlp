"""
Graph Attention Networks in Grap4NLP using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
DGL GCN example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn
"""
import argparse
import os
import time
import warnings
from collections import namedtuple
import dgl
import networkx as nx
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data

from ...data.data import GraphData
from ...modules.graph_embedding_learning.gcn import GCN
from ...modules.utils.generic_utils import EarlyStopping, get_config, grid, print_config

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

warnings.filterwarnings("ignore")


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
        hidden_size,
        output_size,
        direction_option,
        feat_drop=0.0,
        norm="both",
        weight=True,
        bias=True,
        activation=F.elu,
        allow_zero_in_degree=False,
        use_edge_weight=False,
    ):
        super(GNNClassifier, self).__init__()
        self.direction_option = direction_option
        self.model = GCN(
            num_layers,
            input_size,
            hidden_size,
            output_size,
            direction_option=direction_option,
            feat_drop=feat_drop,
            gcn_norm=norm,
            weight=weight,
            bias=bias,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
            use_edge_weight=use_edge_weight,
        )

        if self.direction_option == "bi_sep":
            self.fc = nn.Linear(2 * output_size, output_size)

    def forward(self, graph):
        graph = self.model(graph)
        logits = graph.node_features["node_emb"]
        if self.direction_option == "bi_sep":
            logits = self.fc(F.elu(logits))

        # return logits.log_softmax(dim=-1)
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

    if args.to_undirected:
        inv_edge_index = (g.edges()[1], g.edges()[0])
        g = dgl.add_edges(g, inv_edge_index[0], inv_edge_index[1])
        # adj = to_undirected(edge_index, num_nodes=data[0]['num_nodes'])
        print("convert the input graph to undirected graph")

    # add self loop
    # no duplicate self loop will be added for nodes already having self loops
    new_g = dgl.transform.add_self_loop(g)

    adj = torch.stack(g.edges())

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
        "adj": adj,
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

    adj = data["adj"]

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

    adj = adj.to(device)

    dgl_graph = dgl_graph.to(device)

    dgl_graph.ndata["node_feat"] = features

    # convert DGLGraph to GraphData
    g = GraphData()
    g.from_dgl(dgl_graph)

    g.edge_features["edge_weight"] = (
        torch.Tensor([1] * g.get_edge_num()).float().view(-1, 1).to(device)
    )
    g.edge_features["reverse_edge_weight"] = (
        torch.Tensor([1] * g.get_edge_num()).float().view(-1, 1).to(device)
    )
    g.graph_attributes["adj_t"] = adj
    g = g.to(device)

    # create model
    model = GNNClassifier(
        args.num_layers,
        num_feats,
        args.num_hidden,
        n_classes,
        direction_option=args.direction_option,
        feat_drop=args.feat_drop,
        activation=F.elu,
        use_edge_weight=False,
    )

    # model = GCN(args.num_layers,
    #             num_feats,
    #             args.num_hidden,
    #             n_classes,
    #             args.feat_drop)

    print(model)
    model.to(device)

    if args.early_stop:
        stopper = EarlyStopping("{}.{}".format(args.save_model_path, seed), patience=args.patience)

    loss_fcn = torch.nn.CrossEntropyLoss()
    # loss_fcn = torch.nn.NLLLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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


def multi_run(config):
    config["save_model_path"] = "{}_{}_{}_{}".format(
        config["save_model_path"], config["dataset"], "gcn", config["direction_option"]
    )
    print_config(config)
    config = dict_to_namedtuple(config)

    np.random.seed(config.random_seed)
    scores = []
    for _ in range(config.num_runs):
        seed = np.random.randint(10000)
        scores.append(main(config, seed))

    print(
        "\nTest Accuracy ({} runs): mean {:.4f}, std {:.4f}".format(
            config.num_runs, np.mean(scores), np.std(scores)
        )
    )


def grid_search_main(config):
    print_config(config)
    grid_search_hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            grid_search_hyperparams.append(k)

    best_config = None
    best_score = -1
    configs = grid(config)
    for cnf in configs:
        print("\n")
        for k in grid_search_hyperparams:
            cnf["save_model_path"] += "_{}_{}".format(k, cnf[k])
        print(cnf["save_model_path"])

        score = main(dict_to_namedtuple(cnf), cnf["random_seed"])
        if best_score < score:
            best_score = score
            best_config = cnf
            print("Found a better configuration: {}".format(best_score))

    print("\nBest configuration:")
    for k in grid_search_hyperparams:
        print("{}: {}".format(k, best_config[k]))

    print("Best test score: {}".format(best_score))


def dict_to_namedtuple(data, typename="config"):
    return namedtuple(typename, data.keys())(
        *(
            dict_to_namedtuple(typename + "_" + k, v) if isinstance(v, dict) else v
            for k, v in data.items()
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, help="path to the config file")
    parser.add_argument("--grid_search", action="store_true", help="flag: grid search")
    cfg = vars(parser.parse_args())

    config = get_config(cfg["config"])
    if cfg["grid_search"]:
        grid_search_main(config)
    else:
        multi_run(config)
