import torch
import dgl
import time
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from rgcn import RGCN
from dgl.data.rdf import AIFBDataset
from graph4nlp.pytorch.data.data import from_dgl

def load_data(get_norm=False, inv_target=False):
    dataset = AIFBDataset()

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

    # get target category id
    category_id = hg.ntypes.index(category)
    g = hg
    # print(hg.ndata.keys())
    g = dgl.to_homogeneous(hg, edata=edata)
    # print(g.ndata)
    # print(g.edata)
    # Rename the fields as they can be changed by for example NodeDataLoader
    # g.ndata['ntype'] = g.ndata.pop(dgl.NTYPE)
    # g.ndata['type_id'] = g.ndata.pop(dgl.NID)
    # g.is_homogeneous=False
    node_ids = torch.arange(g.num_nodes())

    # find out the target node ids in g
    # loc = (g.ndata['ntype'] == category_id)
    loc = (g.ndata['_TYPE'] == category_id)
    target_idx = node_ids[loc]

    if inv_target:
        # Map global node IDs to type-specific node IDs. This is required for
        # looking up type-specific labels in a minibatch
        inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64)
        inv_target[target_idx] = torch.arange(0, target_idx.shape[0],
                                           dtype=inv_target.dtype)
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx, inv_target
    else:
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx

if __name__ == "__main__":
    dataset = AIFBDataset()
    hg = dataset[0]

    num_layers = 3
    hidden_size = 64
    dropout = 0.1
    use_self_loop = False
    num_bases = 1
    num_epochs = 5

    g, num_rels, num_classes, labels, train_idx, test_idx, target_idx = load_data(get_norm=True)

    # g.ntypes = 0
    graph = from_dgl(g, is_hetero=False)
    # graph.ntypes=0
    num_nodes = graph.get_node_num()
    model = RGCN(num_hidden_layers=num_layers, 
                 input_size=num_nodes,
                 hidden_size=hidden_size,
                 output_size=num_classes,
                 num_rels=num_rels,
                 num_bases=num_bases,
                 use_self_loop=use_self_loop,
                 dropout = dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()

    for epoch in range(100):
        logits = model(graph).node_features["node_emb"]
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx].argmax(dim=1), labels[train_idx]).item()
        print("Epoch {:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
            epoch, train_acc, loss.item()))
    print()

    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     t0 = time.time()
    #     logits = model(graph)
    #     logits = logits[target_idx]
    #     loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    #     t1 = time.time()
    #     loss.backward()
    #     optimizer.step()
    #     t2 = time.time()

    #     forward_time.append(t1 - t0)
    #     backward_time.append(t2 - t1)
    #     print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
    #           format(epoch, forward_time[-1], backward_time[-1]))
    #     train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
    #     val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
    #     val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
    #     print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
    #           format(train_acc, loss.item(), val_acc, val_loss.item()))
    # print()

    # model.eval()
    # logits = model.forward(g, feats, edge_type, edge_norm)
    # logits = logits[target_idx]
    # test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    # test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    # print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    # print()

    # print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    # print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
