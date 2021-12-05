"""
.. currentmodule:: dgl

Graph Classification Tutorial
=============================

**Author**: `Mufei Li <https://github.com/mufeili>`_,
`Minjie Wang <https://jermainewang.github.io/>`_,
`Zheng Zhang <https://shanghai.nyu.edu/academics/faculty/directory/zheng-zhang>`_.

In this tutorial, you learn how to use DGL to batch multiple graphs of variable size and shape. The
tutorial also demonstrates training a graph neural network for a simple graph classification task.

Graph classification is an important problem
with applications across many fields, such as bioinformatics, chemoinformatics, social
network analysis, urban computing, and cybersecurity. Applying graph neural
networks to this problem has been a popular approach recently.
This can be seen in the following reserach references:
`Ying et al., 2018 <https://arxiv.org/abs/1806.08804>`_,
`Cangea et al., 2018 <https://arxiv.org/abs/1811.01287>`_,
`Knyazev et al., 2018 <https://arxiv.org/abs/1811.09595>`_,
`Bianchi et al., 2019 <https://arxiv.org/abs/1901.01343>`_,
`Liao et al., 2019 <https://arxiv.org/abs/1901.01484>`_,
`Gao et al., 2019 <https://openreview.net/forum?id=HJePRoAct7>`_).

"""

###############################################################################
# Simple graph classification task
# --------------------------------
# In this tutorial, you learn how to perform batched graph classification
# with DGL. The example task objective is to classify eight types of topologies shown here.
#
# .. image:: https://data.dgl.ai/tutorial/batch/dataset_overview.png
#     :align: center
#
# Implement a synthetic dataset :class:`data.MiniGCDataset` in DGL. The dataset has eight
# different types of graphs and each class has the same number of graph samples.
import argparse
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import MiniGCDataset
from torch.utils.data import DataLoader

from ...data.data import from_dgl, to_batch
from ...modules.graph_embedding_learning.gat import GAT
from ...modules.prediction.classification.graph_classification import FeedForwardNN

# import matplotlib.pyplot as plt
# import networkx as nx


# fig, ax = plt.subplots()
# nx.draw(graph.to_networkx(), ax=ax)
# ax.set_title('Class: {:d}'.format(label))
# plt.show()

###############################################################################
# Form a graph mini-batch
# -----------------------
# To train neural networks efficiently, a common practice is to batch
# multiple samples together to form a mini-batch. Batching fixed-shaped tensor
# inputs is common. For example, batching two images of size 28 x 28
# gives a tensor of shape 2 x 28 x 28. By contrast, batching graph inputs
# has two challenges:
#
# * Graphs are sparse.
# * Graphs can have various length. For example, number of nodes and edges.
#
# To address this, DGL provides a :func:`dgl.batch` API. It leverages the idea that
# a batch of graphs can be viewed as a large graph that has many disjointed
# connected components. Below is a visualization that gives the general idea.
#
# .. image:: https://data.dgl.ai/tutorial/batch/batch.png
#     :width: 400pt
#     :align: center
#
# Define the following ``collate`` function to form a mini-batch from a given
# list of graph and label pairs.


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


###############################################################################
# The return type of :func:`dgl.batch` is still a graph. In the same way,
# a batch of tensors is still a tensor. This means that any code that works
# for one graph immediately works for a batch of graphs. More importantly,
# because DGL processes messages on all nodes and edges in parallel, this greatly
# improves efficiency.
#
# Graph classifier
# ----------------
# Graph classification proceeds as follows.
#
# .. image:: https://data.dgl.ai/tutorial/batch/graph_classifier.png
#
# From a batch of graphs, perform message passing and graph convolution
# for nodes to communicate with others. After message passing, compute a
# tensor for graph representation from node (and edge) attributes. This step might
# be called readout or aggregation. Finally, the graph
# representations are fed into a classifier :math:`g` to predict the graph labels.
#
# Graph convolution layer can be found in the ``dgl.nn.<backend>`` submodule.


###############################################################################
# Readout and classification
# --------------------------
# For this demonstration, consider initial node features to be their degrees.
# After two rounds of graph convolution, perform a graph readout by averaging
# over all node features for each graph in the batch.
#
# .. math::
#
#    h_g=\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}h_{v}
#
# In DGL, :func:`dgl.mean_nodes` handles this task for a batch of
# graphs with variable size. You then feed the graph representations into a
# classifier with one linear layer to obtain pre-softmax logits.


class GNNClassifier(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        num_heads,
        num_out_heads,
        direction_option,
        graph_pooling,
        feat_drop=0.6,
        attn_drop=0.6,
        negative_slope=0.2,
        residual=False,
        activation=F.elu,
    ):
        super(GNNClassifier, self).__init__()
        self.direction_option = direction_option
        heads = [num_heads] * (num_layers - 1) + [num_out_heads]
        self.gnn = GAT(
            num_layers,
            input_size,
            hidden_size,
            hidden_size,
            heads,
            direction_option=direction_option,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
        )

        self.clf = FeedForwardNN(
            2 * hidden_size if self.direction_option == "bi_sep" else hidden_size,
            output_size,
            [hidden_size],
            graph_pool_type=graph_pooling,
        )

    def forward(self, batched_graph):
        batched_graph = self.gnn(batched_graph)
        batched_graph = self.clf(batched_graph)
        logits = batched_graph.graph_attributes["logits"]

        return logits


def prepare_batched_graph(dgl_graph):
    # Use node degree as the initial node feature. For undirected graphs,
    # the in-degree is the same as the out_degree.
    node_feat = dgl_graph.in_degrees().view(-1, 1).float()
    dgl_graph.ndata["node_feat"] = node_feat
    g_list = dgl.unbatch(dgl_graph)
    bg = to_batch([from_dgl(g) for g in g_list])
    return bg


###############################################################################
# Setup and training
# ------------------
# Create a synthetic dataset of :math:`400` graphs with :math:`10` ~
# :math:`20` nodes. :math:`320` graphs constitute a training set and
# :math:`80` graphs constitute a test set.


def main(args):
    # A dataset with 80 samples, each graph is
    # of size [10, 20]
    dataset = MiniGCDataset(80, 10, 20)
    graph, label = dataset[0]

    # Create training and test sets.
    trainset = MiniGCDataset(320, 10, 20)
    testset = MiniGCDataset(80, 10, 20)
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    num_feats = 1
    num_classes = 8
    # Create model
    model = GNNClassifier(
        args.num_layers,
        num_feats,
        args.num_hidden,
        num_classes,
        args.num_heads,
        args.num_out_heads,
        direction_option=args.direction_option,
        graph_pooling=args.graph_pooling,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=args.negative_slope,
        residual=args.residual,
        activation=F.elu,
    )

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    epoch_losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        for bg, label in data_loader:
            bg = prepare_batched_graph(bg)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= len(data_loader)
        print("Epoch {}, loss {:.4f}".format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    ###############################################################################
    # The learning curve of a run is presented below.

    # plt.title('cross entropy averaged over minibatches')
    # plt.plot(epoch_losses)
    # plt.show()

    ###############################################################################
    # The trained model is evaluated on the test set created. To deploy
    # the tutorial, restrict the running time to get a higher
    # accuracy (:math:`80` % ~ :math:`90` %) than the ones printed below.

    model.eval()
    # Convert a list of tuples to two lists
    test_X, test_Y = map(list, zip(*testset))
    test_bg = prepare_batched_graph(dgl.batch(test_X))
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    print(
        "Accuracy of sampled predictions on the test set: {:.4f}%".format(
            (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100
        )
    )
    print(
        "Accuracy of argmax predictions on the test set: {:4f}%".format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100
        )
    )


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="GAT")
    # register_data_args(parser)
    parser.add_argument(
        "--graph-pooling",
        type=str,
        default="max_pool",
        help="graph pooling (`avg_pool`, `max_pool`)",
    )
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
    parser.add_argument("--num-hidden", type=int, default=8, help="number of hidden units")
    parser.add_argument(
        "--residual", action="store_true", default=False, help="use residual connection"
    )
    parser.add_argument("--in-drop", type=float, default=0.6, help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.6, help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
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
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="number of epochs to train (default: 100)"
    )
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers (default: 0)")
    parser.add_argument(
        "--dataset", type=str, default="ogbg-molhiv", help="dataset name (default: ogbg-molhiv)"
    )
    parser.add_argument(
        "--feature", type=str, default="full", help="full feature or simple feature"
    )
    parser.add_argument(
        "--filename", type=str, default="", help="filename to output result (default: )"
    )
    args = parser.parse_args()

    main(args)
