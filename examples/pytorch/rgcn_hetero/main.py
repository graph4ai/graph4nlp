"""Deprecated RGCN code on Heterogeneous graph due to 
the lack of support from data structure. The following supports
are needed (but not limit to):
- Redefine the feature data structure of node/edge
    - Index node/edge ids by their type.
    - Enable type indexed features.
- Make corresponding changes on views.
- Make corresponding changes on set/get features functions.

This example bypasses it by storing the features in the model
itself. It is a code trick and therefore not recommended to
the user.
"""
import argparse
import torch
import dgl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from rgcn_hetero import RGCNHetero
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from graph4nlp.pytorch.data.data import from_dgl

# Load dataset 
# Reference: dgl/examples/pytorch/rgcn/entity_utils.py (https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/entity_utils.py)
def load_data(data_name='aifb', get_norm=False):
    if data_name == 'aifb':
        dataset = AIFBDataset()
    elif data_name == 'mutag':
        dataset = MUTAGDataset()
    elif data_name == 'bgs':
        dataset = BGSDataset()
    else:
        dataset = AMDataset()
    # Load hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    labels = hg.nodes[category].data.pop('labels')
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    if get_norm:
        # Calculate normalization weight for each edge,
        # 1. / d, d is the degree of the destination node
        for cetype in hg.canonical_etypes:
            hg.edges[cetype].data['norm'] = dgl.norm_by_dst(hg, cetype).unsqueeze(1)
        edata = ['norm']
    else:
        edata = None
    category_id = hg.ntypes.index(category)
    hg.ndata.pop('label')
    # g = dgl.to_homogeneous(hg, edata=edata)
    g = hg
    node_ids = torch.arange(g.num_nodes())

    # find out the target node ids in g
    loc = (g.ndata['_TYPE'] == category_id)
    target_idx = node_ids[loc]

    return g, category, num_rels, num_classes, labels, train_idx, test_idx, target_idx


def main(args):
    g, category, num_rels, num_classes, labels, train_idx, test_idx, target_idx = load_data(data_name=args.dataset, get_norm=True)

    graph = from_dgl(g, is_hetero=True)
    num_nodes = graph.get_node_num()
    model = RGCNHetero(num_hidden_layers=args.num_hidden_layers, 
                 input_size=num_nodes,
                 hidden_size=args.hidden_size,
                 output_size=num_classes,
                 rel_names=list(set(g.etypes)),
                 node_types=list(graph.ntypes),
                 num_nodes={nt: len(g.ndata['_ID'][nt]) for nt in graph.ntypes},
                 num_bases=args.num_bases,
                 use_self_loop=args.use_self_loop,
                 gpu=args.gpu,
                 dropout = args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print("start training...")
    model.train()
    for epoch in range(args.num_epochs):
        logits = model(graph)[category]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx].argmax(dim=1), labels[train_idx]).item()
        print("Epoch {:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(epoch, train_acc, loss.item()))
    print()
    # Save Model
    # torch.save(model.state_dict(), "./rgcn_model.pt")
    print("start evaluating...")
    model.eval()
    with torch.no_grad():
        logits = model(graph)[category]
    test_acc = accuracy(logits[test_idx].argmax(dim=1), labels[test_idx]).item()
    print("Test Accuracy: {:.4f}".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for entity classification')
    parser.add_argument("--num-hidden-layers", type=int, default=1,
                        help="number of hidden layers beside input/output layer")
    parser.add_argument("--hidden-size", type=int, default=16,
                        help="dimension of hidden layer")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU device number, -1 for cpu")
    parser.add_argument("--num-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("-d", "--dataset", type=str, default='aifb',
                        choices=['aifb', 'mutag', 'bgs', 'am'],
                        help="dataset to use")
    parser.add_argument("--use-self-loop", type=bool, default=False,
                        help="Consider self-loop edges or not")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Start learning rate")
    parser.add_argument("--wd", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of training epochs")

    args = parser.parse_args()
    print(args)
    main(args)