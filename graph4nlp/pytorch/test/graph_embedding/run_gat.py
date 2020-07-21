"""
Graph Attention Networks in Grap4NLP using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
DGL GAT example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import citation_graph as citegrh

from .utils import EarlyStopping
from ...modules.graph_embedding.gat import GAT


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
    def __init__(self,
                num_layers,
                input_size,
                hidden_size,
                output_size,
                num_heads,
                num_out_heads,
                direction_option,
                feat_drop=0.6,
                attn_drop=0.6,
                negative_slope=0.2,
                residual=False,
                activation=F.elu):
        super(GNNClassifier, self).__init__()
        self.direction_option = direction_option
        heads = [num_heads] * (num_layers - 1) + [num_out_heads]
        self.model = GAT(num_layers,
                    input_size,
                    hidden_size,
                    output_size,
                    heads,
                    direction_option=direction_option,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=activation)

        if self.direction_option == 'bi_sep':
            self.fc = nn.Linear(2 * output_size, output_size)

    def forward(self, graph):
        graph = self.model(graph)
        logits = graph.ndata['node_emb']
        if self.direction_option == 'bi_sep':
            logits = self.fc(F.elu(logits))

        return logits


def main(args, seed):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
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
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # Configure
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        print('[ Using CUDA ]')
        device = torch.device('cuda' if args.gpu < 0 else 'cuda:%d' % args.gpu)
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device('cpu')

    if not args.no_cuda and torch.cuda.is_available():
        print('[ Use CUDA ]')
        cuda = True
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
    else:
        cuda = False

    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()
    g.ndata['node_feat'] = features
    # create model
    model = GNNClassifier(args.num_layers,
                    num_feats,
                    args.num_hidden,
                    n_classes,
                    args.num_heads,
                    args.num_out_heads,
                    direction_option=args.direction_option,
                    feat_drop=args.in_drop,
                    attn_drop=args.attn_drop,
                    negative_slope=args.negative_slope,
                    residual=args.residual,
                    activation=F.elu)


    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, g, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

    return acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--num-runs", type=int, default=5,
                        help="number of runs")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="use CPU")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--direction-option", type=str, default='uni',
                        help="direction type (`uni`, `bi_fuse`, `bi_sep`)")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    np.random.seed(123)
    scores = []
    for _ in range(args.num_runs):
        seed = np.random.randint(10000)
        scores.append(main(args, seed))

    print("\nTest Accuracy ({} runs): mean {:.4f}, std {:.4f}".format(args.num_runs, np.mean(scores), np.std(scores)))

