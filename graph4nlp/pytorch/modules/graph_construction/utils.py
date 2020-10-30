from scipy import sparse

from ...data.data import GraphData

CORENLP_TIMEOUT_SIGNATURE = 'CoreNLP request timed out. Your document may be too long.'


def convert_adj_to_graph(adj, reverse_adj, mask_off_val):
    """Convert adjacency matrix to GraphData
    """
    binarized_adj = sparse.coo_matrix(adj.detach().cpu().numpy() != mask_off_val)
    graph_data = GraphData()
    graph_data.from_scipy_sparse_matrix(binarized_adj)
    edge_weight = adj[adj != mask_off_val]
    reverse_edge_weight = reverse_adj[reverse_adj != mask_off_val]

    graph_data.edge_features['edge_weight'] = edge_weight
    graph_data.edge_features['reverse_edge_weight'] = reverse_edge_weight

    return graph_data
