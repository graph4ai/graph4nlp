import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import random
import argparse
import typing as tp
from torchmetrics.functional import accuracy

from ...data.data import GraphData, from_dgl, from_batch, to_batch
from ...modules.graph_embedding_learning.rgcn import RGCNLayer
from ...modules.utils.generic_utils import get_config

from typing import List

import pickle as pkl

# Load dataset
# Reference: dgl/examples/pytorch/rgcn/entity_utils.py
# (https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/entity_utils.py)
def load_data(data_name="aifb", get_norm=False, inv_target=False):
    print(f"Loading dataset: {data_name}")
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

    # if get_norm:
    #     # Calculate normalization weight for each edge,
    #     # 1. / d, d is the degree of the destination node
    #     for cetype in hg.canonical_etypes:
    #         hg.edges[cetype].data["norm"] = dgl.norm_by_dst(hg, cetype).unsqueeze(1)
    #     edata = ["norm"]
    # else:
    #     edata = None
    
    category_id = hg.ntypes.index(category)
    g = hg
    
    node_ids = torch.arange(g.num_nodes())
    
    return g, num_rels, num_classes, labels, train_idx, test_idx, category

def set_seed(seed: int):
    """Set random seed for python, numpy, torch and cuda.

    Parameters
    ----------
    seed : int
        the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_heterograph(g1: dgl.DGLGraph, g2: GraphData):
    # TEST: node/edge num, node/edge features of the two dgl graphs
    assert g2.num_nodes() == g1.num_nodes()
    assert g2.num_edges() == g1.num_edges()
    for ntype in g1.ntypes:
        assert torch.equal(g2.nodes[ntype].data['_ID'], g1.nodes[ntype].data['_ID'])
    for etype in g1.canonical_etypes:
        # assert torch.equal(new_dglg.edges[etype].data["norm"], g.edges[etype].data["norm"])
        assert torch.equal(g2.edges[etype].data["_ID"], g1.edges[etype].data["_ID"])

def check_from_to_batch(g_list: List[dgl.DGLGraph]):
    batch_g = to_batch([from_dgl(g_i) for g_i in g_list])
    new_g_list = from_batch(batch_g)

    assert len(g_list) == len(new_g_list)
    for index in range(len(g_list)):
        check_heterograph(g_list[index], new_g_list[index].to_dgl())
    print("Check `from_batch` and `to_batch` ok...")

def check_from_to_dgl(g1: dgl.DGLGraph):
    graph = from_dgl(g1)
    g2 = graph.to_dgl()
    check_heterograph(g1, g2)
    print("Check `from_dgl` and `to_dgl` ok...")

def main(config):
    set_seed(42)
    g, num_rels, num_classes, labels, train_idx, test_idx, category = load_data(
        data_name="am", get_norm=True
    )

    g1, num_rels, num_classes, labels, train_idx, test_idx, category = load_data(
        data_name="aifb", get_norm=True
    )

    g2, num_rels, num_classes, labels, train_idx, test_idx, category = load_data(
        data_name="mutag", get_norm=True
    )

    g3, num_rels, num_classes, labels, train_idx, test_idx, category = load_data(
        data_name="bgs", get_norm=True
    )

    g_list = [g1, g2, g3, g]

    for g_i in g_list:
        check_from_to_dgl(g_i)
    
    check_from_to_batch(g_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, help="path to the config file")
    parser.add_argument("--grid_search", action="store_true", help="flag: grid search")
    cfg = vars(parser.parse_args())
    config = get_config(cfg["config"])
    print(config)
    main(config)
