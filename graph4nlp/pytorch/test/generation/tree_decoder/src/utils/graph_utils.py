import json
import numpy as np
import networkx as nx
from collections import OrderedDict

word_size_max = 1
sample_size_per_layer = 10


def read_graph_data(input_file):
    # transform the keys to string
    graphs_new = []
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            graph_i = {}
            graph_i['g_ids'] = jo['g_ids']
            graph_i['g_ids_features'] = jo['g_ids_features']
            graph_i['g_adj'] = jo['g_adj']
            graphs_new.append(graph_i)
    return graphs_new


def cons_batch_graph(graphs):
    g_ids = {}
    g_ids_features = {}
    g_fw_adj = {}
    g_bw_adj = {}
    g_nodes = []
    for g in graphs:
        ids = g['g_ids']
        id_adj = g['g_adj']
        features = g['g_ids_features']
        nodes = []

        # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
        # used in the creation of fw_adj and bw_adj

        id_gid_map = {}
        offset = len(g_ids.keys())
        for id in ids:
            id = int(id)
            g_ids[offset + id] = len(g_ids.keys())
            g_ids_features[offset + id] = features[str(id)]
            id_gid_map[id] = offset + id
            nodes.append(offset + id)
        g_nodes.append(nodes)

        for id in id_adj:
            adj = id_adj[id]
            id = int(id)
            g_id = id_gid_map[id]
            if g_id not in g_fw_adj:
                g_fw_adj[g_id] = []
            for t in adj:
                t = int(t)
                g_t = id_gid_map[t]
                g_fw_adj[g_id].append(g_t)
                if g_t not in g_bw_adj:
                    g_bw_adj[g_t] = []
                g_bw_adj[g_t].append(g_id)

    node_size = len(g_ids.keys())
    for id in range(node_size):
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        if id not in g_bw_adj:
            g_bw_adj[id] = []

    graph = {}
    graph['g_ids'] = g_ids
    graph['g_ids_features'] = g_ids_features
    graph['g_nodes'] = g_nodes
    graph['g_fw_adj'] = g_fw_adj
    graph['g_bw_adj'] = g_bw_adj
    return graph


def vectorize_batch_graph(graph, word_manager):
    # vectorize the graph feature and normalize the adj info
    id_features = graph['g_ids_features']
    gv = {}
    nv = []
    word_max_len = 0
    for id in id_features:
        feature = id_features[id]
        word_max_len = max(word_max_len, len(feature.split()))
    word_max_len = min(word_max_len,  word_size_max)

    for id in graph['g_ids_features']:
        feature = graph['g_ids_features'][id]
        fv = []
        for token in feature.split():
            if len(token) == 0:
                continue
            # if token in word_idx:
            #     fv.append(word_idx[token])
            # else:
            #     fv.append(word_idx['<U>'])
            fv.append(word_manager.get_symbol_idx(token))
        for _ in range(word_max_len - len(fv)):
            fv.append(0)
        fv = fv[:word_max_len]
        nv.append(fv)

    nv.append([0 for temp in range(word_max_len)])
    gv['g_ids_features'] = np.array(nv)

    g_fw_adj = graph['g_fw_adj']
    g_fw_adj_v = []

    degree_max_size = 0
    for id in g_fw_adj:
        degree_max_size = max(degree_max_size, len(g_fw_adj[id]))

    g_bw_adj = graph['g_bw_adj']
    for id in g_bw_adj:
        degree_max_size = max(degree_max_size, len(g_bw_adj[id]))

    degree_max_size = min(degree_max_size, sample_size_per_layer)

    for id in g_fw_adj:
        adj = g_fw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_fw_adj.keys()))
        adj = adj[:degree_max_size]
        g_fw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

    g_bw_adj_v = []
    for id in g_bw_adj:
        adj = g_bw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_bw_adj.keys()))
        adj = adj[:degree_max_size]
        g_bw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

    gv['g_ids'] = graph['g_ids']
    gv['g_nodes'] = np.array(graph['g_nodes'])
    gv['g_bw_adj'] = np.array(g_bw_adj_v)
    gv['g_fw_adj'] = np.array(g_fw_adj_v)
    return gv
