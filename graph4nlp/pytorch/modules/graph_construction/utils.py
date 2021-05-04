from copy import copy
from scipy import sparse
import torch

from ...data.data import GraphData

CORENLP_TIMEOUT_SIGNATURE = 'CoreNLP request timed out. Your document may be too long.'

def convert_adj_to_graph(graph, adj, reverse_adj, mask_off_val):

    slides = (adj != mask_off_val).nonzero(as_tuple=False)
    from copy import deepcopy
    batch_nodes_tensor = torch.Tensor([0] + graph._batch_num_nodes).to(slides.device)
    batch_prefix = batch_nodes_tensor.view(-1, 1).expand(-1, batch_nodes_tensor.shape[0])

    batch_prefix = batch_prefix.triu().long().sum(0)


    # for i in range(1, len(graph._batch_num_nodes)):
    #     cnt = batch_nodes[-1] + graph._batch_num_nodes[i-1]
    #     batch_nodes.append(cnt)

    # pre_num = [batch_nodes[i] for i in slides[:, 0]]


    src = slides[:, 1] + batch_prefix.index_select(dim=0, index=slides[:, 0])
    tgt = slides[:, 2] + batch_prefix.index_select(dim=0, index=slides[:, 0])

    graph_data = graph
    graph_data.remove_all_edges() # remove all existing edges
    graph_data.add_edges(src.detach().cpu().numpy().tolist(), tgt.detach().cpu().numpy().tolist())

    value = adj[slides[:, 0], slides[:, 1], slides[:, 2]]
    reverse_value = reverse_adj[slides[:, 0], slides[:, 1], slides[:, 2]]
    graph_data.edge_features['edge_weight'] = value
    graph_data.edge_features['reverse_edge_weight'] = reverse_value

    return graph_data





# def convert_adj_to_graph(graph, adj, reverse_adj, mask_off_val):
#     """Convert adjacency matrix to GraphData
#     """
#     # slides = (adj != mask_off_val).nonzero()
#     # from copy import deepcopy
#     # batch_nodes = [0]


#     # for i in range(1, len(graph._batch_num_nodes)):
#     #     cnt = batch_nodes[-1] + graph._batch_num_nodes[i-1]
#     #     batch_nodes.append(cnt)

#     # # pre_num = [batch_nodes[i] for i in slides[:, 0]]


#     # src = slides[:, 1] + slides[:, 0] * 100
#     # tgt = slides[:, 2] + slides[:, 0]* 100

#     # graph_data = graph
#     # graph_data.add_edges(src.detach().cpu().numpy().tolist(), tgt.detach().cpu().numpy().tolist())

#     # value = adj[slides[:, 0], slides[:, 1], slides[:, 2]]
#     # reverse_value = reverse_adj[slides[:, 0], slides[:, 1], slides[:, 2]]
#     # #graph_data.edge_features['edge_weight'] = value
#     # #graph_data.edge_features['reverse_edge_weight'] = reverse_value

#     # return graph_data


#     slides = (adj != mask_off_val).nonzero()
#     src = slides[:, 0]
#     tgt = slides[:, 1]
#     graph.add_edges(src.detach().cpu().numpy().tolist(), tgt.detach().cpu().numpy().tolist())
#     # value = adj[slides[:, 0], slides[:, 1]]
#     # reverse_value = reverse_adj[slides[:, 0], slides[:, 1]]
#     # graph.edge_features['edge_weight'] = value
#     # graph.edge_features['reverse_edge_weight'] = reverse_value
#     return graph




#     binarized_adj = sparse.coo_matrix(adj.detach().cpu().numpy() != mask_off_val)
#     graph_data = GraphData()
#     graph_data.from_scipy_sparse_matrix(binarized_adj)
#     edge_weight = adj[adj != mask_off_val]
#     reverse_edge_weight = reverse_adj[reverse_adj != mask_off_val]

#     graph_data.edge_features['edge_weight'] = edge_weight
#     graph_data.edge_features['reverse_edge_weight'] = reverse_edge_weight

#     return graph_data
