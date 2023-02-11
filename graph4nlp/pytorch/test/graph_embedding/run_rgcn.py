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

from ...data.data import GraphData, from_dgl
from ...modules.graph_embedding_learning.rgcn import RGCNLayer
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


class MyModel(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        num_rels,
        direction_option=None,
        bias=True,
        activation=None,
        self_loop=True,
        feat_drop=0.0,
        regularizer="none",
        num_bases=4,
        num_nodes=100,
        g: GraphData = None,
    ):
        super(MyModel, self).__init__()
        self.emb = nn.Embedding(num_nodes, hidden_size)
        nn.init.xavier_uniform_(self.emb.weight, gain=nn.init.calculate_gain("relu"))
        self.layer_1 = RGCNLayer(
            input_size,
            hidden_size,
            num_rels=num_rels,
            direction_option=direction_option,
            bias=bias,
            activation=activation,
            self_loop=self_loop,
            feat_drop=feat_drop,
            regularizer=regularizer,
            num_bases=num_bases,
        )
        self.layer_2 = RGCNLayer(
            hidden_size,
            output_size,
            num_rels=num_rels,
            direction_option=direction_option,
            bias=bias,
            activation=activation,
            self_loop=self_loop,
            feat_drop=feat_drop,
            regularizer=regularizer,
            num_bases=num_bases,
        )
        
        self.dgl_g = g.to_dgl()
        self.g = g
        
        # self.feat_dict = feat_dict
    
    @property
    def feat_dict(self):
        node_features = self.emb(torch.IntTensor(list(range(self.g.get_node_num()))).to(self.g.device))
        # Make node feature dictionary
        feat_dict: tp.Dict[str, torch.Tensor] = {}
        node_types = np.array(self.g.ntypes,)
        for i in set(node_types):
            index = torch.tensor(np.where(node_types == i)[0], device=self.g.device)
            feat_dict[i] = torch.index_select(node_features, 0, index)
        return feat_dict

    def forward(self,):
        
        x1 = self.layer_1(self.dgl_g, self.feat_dict)
        x2 = self.layer_2(self.dgl_g, x1)
        # self.feat_dict = x2
        return x2

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

def main(config):
    set_seed(42)
    g, num_rels, num_classes, labels, train_idx, test_idx, category = load_data(
        data_name=config["dataset"], get_norm=True
    )

    # graph = from_dgl(g, is_hetero=False)
    device = "cuda:0"
    
    
    graph = from_dgl(g).to(device)
    new_dglg = graph.to_dgl()
    
    # TEST: node/edge num, node/edge features of the two dgl graphs
    assert new_dglg.num_nodes() == g.num_nodes()
    assert new_dglg.num_edges() == g.num_edges()
    for ntype in g.ntypes:
        assert torch.equal(new_dglg.nodes[ntype].data['_ID'].cpu(), g.nodes[ntype].data['_ID'])
    # for etype in g.etypes:
    #     assert torch.equal(new_dglg.edges[etype].data["norm"], g.edges[etype].data["norm"])
    
    
    
    labels = labels.to(device)
    num_nodes = graph.get_node_num()
    my_model = MyModel(
        num_layers=config["num_hidden_layers"],
        input_size=config["hidden_size"],
        hidden_size=config["hidden_size"],
        output_size=num_classes,
        direction_option=config["direction_option"],
        bias=config["bias"],
        activation=F.relu,
        num_rels=num_rels,
        self_loop=config["self_loop"],
        feat_drop=config["feat_drop"],
        regularizer="basis",
        num_bases=num_rels,
        num_nodes=num_nodes,
        g=graph,
    ).to(device)
    optimizer = torch.optim.Adam(
        my_model.parameters(),
        lr=config["lr"],
        weight_decay=config["wd"],
    )
    print("start training...")
    my_model.train()
    for epoch in range(config["num_epochs"]):
        logits = my_model()[category]
        # logits = logits[train_idx]
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
    my_model.eval()
    with torch.no_grad():
        logits = my_model()['_N']
    # logits = logits[target_idx]
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
