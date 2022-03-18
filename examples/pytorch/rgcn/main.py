import torch
import dgl
import time
import torch.nn.functional as F
from rgcn import RGCN
from dgl.data.rdf import AIFBDataset
from graph4nlp.pytorch.data.data import from_dgl

if __name__ == "__main__":
    dataset = AIFBDataset()
    hg = dataset[0]

    num_layers = 3
    hidden_size = 64
    dropout = 0.1
    use_self_loop = False
    num_bases = -1
    num_epochs = 5

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('labels')

    val_idx = train_idx

    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data['norm'] = norm
    
    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i

    hg.edges
    g = dgl.to_homogeneous(hg, edata=['norm'])
    num_nodes = g.number_of_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes)

    optimizer = torch.optim

    graph = from_dgl(g)
    num_nodes = graph.get_node_num()
    model = RGCN(num_layers=num_layers, 
                 input_size=num_nodes,
                 hidden_size=hidden_size,
                 output_size=num_classes,
                 num_rels=num_rels,
                 num_bases=num_bases,
                 use_self_loop=use_self_loop,
                 dropout = dropout)
    
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(graph)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    logits = model.forward(g, feats, edge_type, edge_norm)
    logits = logits[target_idx]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))