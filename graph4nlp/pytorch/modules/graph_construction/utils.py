import torch

CORENLP_TIMEOUT_SIGNATURE = "CoreNLP request timed out. Your document may be too long."


def convert_adj_to_graph(graph, adj, reverse_adj, mask_off_val):
    slides = (adj != mask_off_val).nonzero(as_tuple=False)

    batch_nodes_tensor = torch.Tensor([0] + graph._batch_num_nodes).to(slides.device)
    batch_prefix = batch_nodes_tensor.view(-1, 1).expand(-1, batch_nodes_tensor.shape[0])

    batch_prefix = batch_prefix.triu().long().sum(0)

    src = slides[:, 1] + batch_prefix.index_select(dim=0, index=slides[:, 0])
    tgt = slides[:, 2] + batch_prefix.index_select(dim=0, index=slides[:, 0])

    graph_data = graph
    graph_data.remove_all_edges()  # remove all existing edges
    graph_data.add_edges(src.detach().cpu().numpy().tolist(), tgt.detach().cpu().numpy().tolist())

    value = adj[slides[:, 0], slides[:, 1], slides[:, 2]]
    reverse_value = reverse_adj[slides[:, 0], slides[:, 1], slides[:, 2]]
    graph_data.edge_features["edge_weight"] = value
    graph_data.edge_features["reverse_edge_weight"] = reverse_value

    return graph_data
