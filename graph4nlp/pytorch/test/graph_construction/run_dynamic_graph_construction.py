import argparse
import os
import time
import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data, register_data_args
from scipy import sparse

from ...modules.graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
    NodeEmbeddingBasedRefinedGraphConstruction,
)
from ...modules.utils.vocab_utils import VocabModel
from ..utils import EarlyStopping


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, g, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(g)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


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


def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sparse.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def get_normalized_init_adj(graph):
    adj = graph.adjacency_matrix_scipy(return_edge_ids=False)
    adj = normalize_sparse_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


class GNNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GNNLayer, self).__init__()
        self.linear_out = nn.Linear(input_size, output_size, bias=False)

    def forward(self, dgl_graph, node_emb):
        with dgl_graph.local_scope():
            dgl_graph.srcdata.update({"ft": node_emb})
            dgl_graph.update_all(fn.u_mul_e("ft", "edge_weight", "m"), fn.sum("m", "ft"))
            agg_vec = dgl_graph.dstdata["ft"]
            new_node_vec = self.linear_out(agg_vec)

            return new_node_vec


class GNNClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, feat_drop=0, layer_drop=0.5, activation=F.relu
    ):
        super(GNNClassifier, self).__init__()
        self.feat_drop = feat_drop
        self.layer_drop = layer_drop
        self.activation = activation
        self.gnn_layer1 = GNNLayer(input_size, hidden_size)
        self.gnn_layer2 = GNNLayer(hidden_size, output_size)

    def forward(self, graph):
        node_feat = graph.ndata["node_feat"]
        node_feat = F.dropout(node_feat, self.feat_drop, training=self.training)

        node_emb = self.activation(self.gnn_layer1(graph, node_feat))
        node_emb = F.dropout(node_emb, self.layer_drop, training=self.training)

        logits = self.gnn_layer2(graph, node_emb)

        return logits


class DynamicGNNClassifier(nn.Module):
    def __init__(
        self,
        word_vocab,
        embedding_styles,
        input_size,
        hidden_size,
        output_size,
        gl_type,
        gl_metric_type="weighted_cosine",
        gl_num_heads=1,
        gl_top_k=None,
        gl_epsilon=None,
        gl_smoothness_ratio=None,
        gl_connectivity_ratio=None,
        gl_sparsity_ratio=None,
        gl_input_size=None,
        gl_hidden_size=None,
        init_adj_alpha=None,
        feat_drop=0,
        layer_drop=0.5,
        activation=F.relu,
        device=None,
    ):
        super(DynamicGNNClassifier, self).__init__()
        self.gl_type = gl_type
        if self.gl_type == "node_emb":
            self.graph_learner = NodeEmbeddingBasedGraphConstruction(
                word_vocab,
                embedding_styles,
                sim_metric_type=gl_metric_type,
                num_heads=gl_num_heads,
                top_k_neigh=gl_top_k,
                epsilon_neigh=gl_epsilon,
                smoothness_ratio=gl_smoothness_ratio,
                connectivity_ratio=gl_connectivity_ratio,
                sparsity_ratio=gl_sparsity_ratio,
                input_size=gl_input_size,
                hidden_size=gl_hidden_size,
                fix_word_emb=True,
                dropout=None,
                device=device,
            )

        elif self.gl_type == "node_edge_emb":
            raise NotImplementedError()
        elif self.gl_type == "node_emb_refined":
            self.graph_learner = NodeEmbeddingBasedRefinedGraphConstruction(
                word_vocab,
                embedding_styles,
                init_adj_alpha,
                sim_metric_type=gl_metric_type,
                num_heads=gl_num_heads,
                top_k_neigh=gl_top_k,
                epsilon_neigh=gl_epsilon,
                smoothness_ratio=gl_smoothness_ratio,
                connectivity_ratio=gl_connectivity_ratio,
                sparsity_ratio=gl_sparsity_ratio,
                input_size=gl_input_size,
                hidden_size=gl_hidden_size,
                fix_word_emb=True,
                dropout=None,
                device=device,
            )
        else:
            raise RuntimeError("Unknown gl_type: {}".format(self.gl_type))

        self.gnn_clf = GNNClassifier(
            input_size,
            hidden_size,
            output_size,
            feat_drop=feat_drop,
            layer_drop=layer_drop,
            activation=activation,
        )

    def forward(self, graph):
        node_feat = graph.ndata["node_feat"]

        if self.gl_type == "node_emb_refined":
            new_graph = self.graph_learner.topology(node_feat, graph.init_adj, node_mask=None)
        else:
            new_graph = self.graph_learner.topology(node_feat, node_mask=None)

        # convert GraphData to DGLGraph
        dgl_graph = new_graph.to_dgl()
        dgl_graph.ndata["node_feat"] = node_feat
        dgl_graph.edata["edge_weight"] = new_graph.edge_features["edge_weight"]
        dgl_graph.graph_reg = new_graph.graph_attributes["graph_reg"]
        logits = self.gnn_clf(dgl_graph)

        return logits, dgl_graph


def main(args, seed):
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

    # load and preprocess dataset
    if args.dataset.startswith("ogbn"):
        # Open Graph Benchmark datasets
        data = prepare_ogbn_graph_data(args)
    else:
        # DGL datasets
        data = prepare_dgl_graph_data(args)

    features, g, train_mask, val_mask, test_mask, labels, num_feats, n_classes, n_edges = (
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

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    g.ndata["node_feat"] = features

    if args.gl_type == "node_emb_refined":
        init_adj = get_normalized_init_adj(g).to(device)
        g.init_adj = init_adj

    raw_text_data = [["I like nlp.", "Same here!"], ["I like graph.", "Same here!"]]
    vocab_model = VocabModel(
        raw_text_data, max_word_vocab_size=None, min_word_vocab_freq=1, word_emb_size=300
    )

    embedding_styles = {
        "word_emb_type": "w2v",
        "node_edge_emb_strategy": "bilstm",
        "seq_info_encode_strategy": "none",
    }

    # create model
    model = DynamicGNNClassifier(
        vocab_model.word_vocab,
        embedding_styles,
        num_feats,
        args.num_hidden,
        n_classes,
        args.gl_type,
        gl_metric_type=args.gl_metric_type,
        gl_num_heads=args.gl_num_heads,
        gl_top_k=args.gl_top_k,
        gl_epsilon=args.gl_epsilon,
        gl_smoothness_ratio=args.gl_smoothness_ratio,
        gl_connectivity_ratio=args.gl_connectivity_ratio,
        gl_sparsity_ratio=args.gl_sparsity_ratio,
        gl_input_size=num_feats,
        gl_hidden_size=args.gl_num_hidden,
        init_adj_alpha=args.init_adj_alpha,
        feat_drop=args.in_drop,
        layer_drop=args.layer_drop,
        activation=F.relu,
        device=device,
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
        logits, dgl_graph = model(g)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        loss += dgl_graph.graph_reg

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
    parser = argparse.ArgumentParser(description="DynamicGraphConstruction")
    register_data_args(parser)
    parser.add_argument("--num-runs", type=int, default=5, help="number of runs")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="use CPU")
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--gl-num-hidden",
        type=int,
        default=16,
        help="number of hidden units for dynamic graph construction",
    )
    parser.add_argument("--gl-top-k", type=int, help="top k for graph sparsification")
    parser.add_argument("--gl-epsilon", type=float, help="epsilon for graph sparsification")
    parser.add_argument(
        "--gl-smoothness-ratio", type=float, help="smoothness ratio for graph regularization loss"
    )
    parser.add_argument(
        "--gl-connectivity-ratio",
        type=float,
        help="connectivity ratio for graph regularization loss",
    )
    parser.add_argument(
        "--gl-sparsity-ratio", type=float, help="sparsity ratio for graph regularization loss"
    )
    parser.add_argument(
        "--gl-num-heads", type=int, default=1, help="num of heads for dynamic graph construction"
    )
    parser.add_argument(
        "--gl-type",
        type=str,
        default="node_emb",
        help=r"dynamic graph construction algorithm type, \
                        {'node_emb', 'node_edge_emb' and 'node_emb_refined'},\
                        default: 'node_emb'",
    )
    parser.add_argument(
        "--gl-metric-type",
        type=str,
        default="weighted_cosine",
        help=r"similarity metric type for dynamic graph construction",
    )
    parser.add_argument(
        "--init-adj-alpha",
        type=float,
        default=0.8,
        help="alpha ratio for combining initial graph adjacency matrix",
    )
    parser.add_argument("--num-hidden", type=int, default=16, help="number of hidden units")
    parser.add_argument("--in-drop", type=float, default=0, help="input feature dropout")
    parser.add_argument("--layer-drop", type=float, default=0.5, help="layer dropout")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
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
        "--save-model-path", type=str, default="ckpt", help="path to the best saved model"
    )
    args = parser.parse_args()
    args.save_model_path = (
        "{}_{}_gl_type_{}_gl_metric_type_{}_gl_heads_{}"
        "_gl_topk_{}_gl_epsilon_{}_smoothness_{}_connectivity_{}"
        "_sparsity_{}_init_adj_alpha_{}"
    ).format(
        args.save_model_path,
        args.dataset,
        args.gl_type,
        args.gl_metric_type,
        args.gl_num_heads,
        args.gl_top_k,
        args.gl_epsilon,
        args.gl_smoothness_ratio,
        args.gl_connectivity_ratio,
        args.gl_sparsity_ratio,
        args.init_adj_alpha,
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
