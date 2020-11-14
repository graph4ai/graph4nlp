import random

import torch

from ...data.data import GraphData
from datetime import datetime
import random

# Test hyperparameters
num_nodes = 100
num_edges = 50

src_list, tgt_list = [], []
for i in range(num_edges):
    src = random.randint(a=0, b=num_nodes - 1)
    tgt = random.randint(a=0, b=num_nodes - 1)
    while tgt == src:
        tgt = random.randint(a=0, b=num_nodes - 1)
    src_list.append(src)
    tgt_list.append(tgt)

# Test build graph and add nodes
g = GraphData()
g.add_nodes(num_nodes)

# Add node attributes
for i in range(num_nodes):
    g.node_attributes[i]['name'] = '*NAMEFIX_{}'.format(i)
    g.node_attributes[i]['value'] = i

# Test add edges
g.add_edges(src_list, tgt_list)

# Test modify and access node features and attributes
emb_layer = torch.nn.Embedding(100000, 10)
g.node_features['node_feat'] = torch.zeros((num_nodes, 10))
adam = torch.optim.Adam(emb_layer.parameters())


# Do some computation here
for i in range(100):
    for i in range(num_nodes):
        g.node_features['node_feat'][i] = emb_layer(torch.tensor([i], dtype=torch.long))

    t0 = datetime.now()
    dgl_g = g.to_dgl()
    print('Time elapsed for convertion = {}.'.format(datetime.now() - t0))

    loss = -torch.nn.functional.cosine_similarity(dgl_g.ndata['node_feat'].sum(dim=-1), torch.ones(num_nodes), dim=0)
    adam.zero_grad()
    loss.backward(retain_graph=True)
    adam.step()
    print('loss = {}'.format(loss))
pass

# # User Code Proposal 1
# graph_data = graph4nlp.graph_construction.CTGraph(raw_data)
# new_graph_data = graph4nlp.graph_construction.GAT(graph_data)
# result = graph4nlp.prediction.NodeClassfier(new_graph_data)
#
# # User Code Proposal 2
# graph_data = graph4nlp.graph_construction.CTGraph(raw_data)
# dgl_graphs = graph_data.to_dgl()
# features = dgl_graphs.ndata['node_feat']  # The key here ('node_feat') can be anything
# new_feats = graph4nlp.graph_construction.GAT(dgl_graphs, features)
# graph_data.node_features['node_emb'] = new_feats
#
#
# # Library code
# class GAT(SomeBase):
#     def __init__(self, num_layers):
#         self.gat_layers = nn.ModuleList
#
#     def forward(self, graph_data: GraphData):
#         dgl_g = graph_data.to_dgl()
#         features = dgl_g.ndata['node_feat']  # 'node_emb' is the reserved key
#         new_features = self.gat_layers(dgl_g, features)
#         graph_data.node_features['node_emb'] = new_features
#         return graph_data
#
#
# class NodeClassifier(AnotherBase):
#     def __init__(self):
#         self.node_classification_layer = NodeClassificationLayer()
#
#     def forward(self, graph_data: GraphData):
#         node_emb = graph_data.node_features['node_emb']
#         return self.node_classification_layer(node_emb)
#
#
# # User implement another model
# class MyGAT(SomeBase):
#     def forward(self, graph_data: GraphData):
#         dgl_g = graph_data.to_dgl()
#         # Extract features here
#         features_1 = dgl_g.ndata['my_feat']
#         features_2 = dgl_g.ndata['another_feat']
#         new_features_1, new_features_2 = MyGATLayer(dgl_g, features_1, features_2)
#         graph_data.node_features['my_node_emb'] = new_features_1
#         graph_data.node_features['node_emb_2'] = new_features_2
#
#
# pass
