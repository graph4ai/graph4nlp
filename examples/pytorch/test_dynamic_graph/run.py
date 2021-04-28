import torch
from torch import nn

from graph4nlp.pytorch.data.data import GraphData, to_batch
from graph4nlp.pytorch.modules.graph_construction.utils import convert_adj_to_graph



class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, graph, raw_adj):
        raw_adj = self.linear(raw_adj)
        mask = (raw_adj > 0).detach().float()
        raw_adj = raw_adj * mask + 0 * (1 - mask)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=1e-10)
        reverse_adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-2, keepdim=True), min=1e-10)
        print(adj)

        graph = convert_adj_to_graph(graph, adj, reverse_adj, 0)
        loss = torch.mean(graph.edge_features['edge_weight']) + torch.mean(graph.edge_features['reverse_edge_weight'])

        return loss


a1 = GraphData()
a1.add_nodes(1)
a1.add_nodes(2)
a1.add_nodes(3)
a2 = GraphData()
a2.add_nodes(4)
a2.add_nodes(5)
a2.add_nodes(6)
graph = to_batch([a1, a2])
raw_adj = torch.randn(2, 3, 3)

model = Model(3)
loss = model(graph, raw_adj)
loss.backward()

# adj.retain_grad()
# reverse_adj.retain_grad()
# graph.edge_features['edge_weight'].retain_grad()
# graph.edge_features['reverse_edge_weight'].retain_grad()

print('grad of model weight')
print(model.linear.weight.grad)
print(model.linear.bias.grad)
