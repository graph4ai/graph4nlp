from ...data.data import GraphData, from_batch, to_batch
import random
import torch

num_nodes = 50
num_edges = 50
num_graphs = 10

graph_list = []

for i in range(num_graphs):
    g = GraphData()
    g.add_nodes(num_nodes)
    src_list, tgt_list = [], []

    for i in range(num_edges):
        src = random.randint(a=0, b=num_nodes - 1)
        tgt = random.randint(a=0, b=num_nodes - 1)
        while tgt == src:
            tgt = random.randint(a=0, b=num_nodes - 1)
        src_list.append(src)
        tgt_list.append(tgt)

    for i in range(num_nodes):
        g.node_attributes[i]['name'] = '*NAMEFIX_{}'.format(i)
        g.node_attributes[i]['value'] = i
    g.add_edges(src_list, tgt_list)

    g.node_features['node_feat'] = torch.rand((50, 10))
    g.edge_features['edge_feat'] = torch.rand((50, 10))

    graph_list.append(g)

batch = to_batch(graph_list)
graph_list_2 = from_batch(batch)
batch.adj_matrix()
batch.scipy_sparse_adj()
eyes = torch.eye(10)
g1 = GraphData()
g1.from_dense_adj(eyes)
import numpy as np
a = np.array(list(range(10)))
import scipy.sparse.coo
coo = scipy.sparse.coo.coo_matrix((np.ones(10), (a, a)), shape=(10, 10))
g2 = GraphData()
g2.from_scipy_sparse_matrix(coo)
pass
