import argparse
import dgl
import torch
import torch.nn.functional as F
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset

from torchmetrics.functional import accuracy

from ...data.data import from_dgl
from ...modules.graph_embedding_learning.rgcn import RGCN
from ...modules.utils.generic_utils import get_config


# Load dataset
# Reference: dgl/examples/pytorch/rgcn/entity_utils.py
# (https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/entity_utils.py)
def load_data(data_name="aifb", get_norm=False, inv_target=False):
    if data_name == "aifb":
        dataset = AIFBDataset()
        # Test Accuracy:
        # 0.9444, 0.8889, 0.9722, 0.9167, 0.9444 without enorm.
        # 0.8611, 0.8889, 0.8889, 0.8889, 0.8333
        # avg: 0.93332 (without enorm)
        # avg: 0.87222
        # DGL: 0.8889, 0.8889, 0.8056, 0.8889, 0.8611
        # DGL avg: 0.86668
        # paper: 0.9583
        # note: Could stuck at Local minimum of train loss between 0.2-0.35.
    elif data_name == "mutag":
        dataset = MUTAGDataset()
        # Test Accuracy:
        # 0.6912, 0.7500, 0.7353, 0.6324, 0.7353
        # avg: 0.68884
        # DGL: 0.6765, 0.7059, 0.7353, 0.6765, 0.6912
        # DGL avg: 0.69724
        # paper: 0.7323
        # note: Could stuck at local minimum of train acc: 0.3897 & loss 0.6931
    elif data_name == "bgs":
        dataset = BGSDataset()
        # Test Accuracy:
        # 0.8966, 0.9310, 0.8966, 0.7931, 0.8621
        # avg: 0.87588
        # DGL: 0.7931, 0.9310, 0.8966, 0.8276, 0.8966
        # DGL avg: 0.86898
        # paper: 0.8310
        # note: Could stuck at local minimum of train acc: 0.6325 & loss: 0.6931
    else:
        dataset = AMDataset()
        # Test Accuracy:
        # 0.7525, 0.7374, 0.7424, 0.7424, 0.7424
        # avg: 0.74342
        # DGL: 0.7677, 0.7677, 0.7323, 0.7879, 0.7677
        # DGL avg: 0.76466
        # paper: 0.8929
        # note: args.hidden_size is 10.
        # Could stuck at local minimum of train loss: 0.3-0.5

    # Load hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    labels = hg.nodes[category].data.pop("labels")
    train_mask = hg.nodes[category].data.pop("train_mask")
    test_mask = hg.nodes[category].data.pop("test_mask")
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    if get_norm:
        # Calculate normalization weight for each edge,
        # 1. / d, d is the degree of the destination node
        for cetype in hg.canonical_etypes:
            hg.edges[cetype].data["norm"] = dgl.norm_by_dst(hg, cetype).unsqueeze(1)
        edata = ["norm"]
    else:
        edata = None
    category_id = hg.ntypes.index(category)
    g = dgl.to_homogeneous(hg, edata=edata)
    node_ids = torch.arange(g.num_nodes())

    # find out the target node ids in g
    loc = g.ndata["_TYPE"] == category_id
    target_idx = node_ids[loc]

    if inv_target:
        # Map global node IDs to type-specific node IDs. This is required for
        # looking up type-specific labels in a minibatch
        inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64)
        inv_target[target_idx] = torch.arange(0, target_idx.shape[0], dtype=inv_target.dtype)
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx, inv_target
    else:
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx


def main(config):
    g, num_rels, num_classes, labels, train_idx, test_idx, target_idx = load_data(
        data_name=config["dataset"], get_norm=True
    )

    # graph = from_dgl(g, is_hetero=False)
    graph = from_dgl(g)
    num_nodes = graph.get_node_num()
    emb = torch.nn.Embedding(num_nodes, config["hidden_size"])
    # emb.requires_grad = True
    graph.node_features["node_feat"] = emb.weight

    model = RGCN(
        num_layers=config["num_hidden_layers"],
        input_size=config["hidden_size"],
        hidden_size=config["hidden_size"],
        output_size=num_classes,
        direction_option=config["direction_option"],
        regularizer="basis",
        bias=True,
        activation=F.relu,
        num_rels=num_rels,
        num_bases=config["num_bases"],
        self_loop=config["self_loop"],
        feat_drop=config["feat_drop"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    print("start training...")
    model.train()
    for epoch in range(config["num_epochs"]):
        logits = model(graph).node_features["node_emb"]
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx].argmax(dim=1), labels[train_idx]).item()
        print(
            "Epoch {:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
                epoch, train_acc, loss.item()
            )
        )
    print()
    # Save Model
    # torch.save(model.state_dict(), "./rgcn_model.pt")
    print("start evaluating...")
    model.eval()
    with torch.no_grad():
        logits = model(graph).node_features["node_emb"]
    logits = logits[target_idx]
    test_acc = accuracy(logits[test_idx].argmax(dim=1), labels[test_idx]).item()
    print("Test Accuracy: {:.4f}".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, help="path to the config file")
    parser.add_argument("--grid_search", action="store_true", help="flag: grid search")
    cfg = vars(parser.parse_args())
    config = get_config(cfg["config"])
    print(config)
    main(config)
