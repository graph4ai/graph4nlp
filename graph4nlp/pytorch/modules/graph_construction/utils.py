from scipy import sparse
import torch
import dgl
from dgl.nn.pytorch.softmax import edge_softmax

from ..utils.torch_utils import to_cuda
from ..utils.constants import INF


def sparsify_graph(attention, top_k, mask_off_val, device=None):
    """Sparsifying a graph
    TODO: support more ways for graph sparsification
    """
    adj = build_knn_neighbourhood(attention, top_k, mask_off_val, device=device)
    return adj

def build_knn_neighbourhood(attention, top_k, mask_off_val, device=None):
    top_k = min(top_k, attention.size(-1))
    knn_val, knn_ind = torch.topk(attention, top_k, dim=-1)
    weighted_adjacency_matrix = to_cuda((mask_off_val * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), device)

    return weighted_adjacency_matrix

def convert_adj_to_dgl_graph(adj, mask_off_val, use_edge_softmax=False):
    """Convert adjacency matrix to DGLGraph
    """
    binarized_adj = sparse.coo_matrix(adj.detach().cpu().numpy() != mask_off_val)
    dgl_graph = dgl.DGLGraph(binarized_adj)
    edge_weight = adj[adj != mask_off_val]

    if use_edge_softmax:
        edge_weight = edge_softmax(dgl_graph, edge_weight)

    dgl_graph.edata['a'] = edge_weight

    return dgl_graph
