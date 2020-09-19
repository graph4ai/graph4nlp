import pytest
import torch
import torch.nn as nn

from ...data.data import GraphData, from_batch, to_batch, from_dgl
from ...data.utils import EdgeNotFoundException


def fail_here():
    raise Exception('The above line of code shouldn\'t be executed normally')


def test_add_nodes():
    g = GraphData()
    try:
        g.add_nodes(-1)
        fail_here()
    except AssertionError:
        pass

    g.add_nodes(10)
    assert g.get_node_num() == 10
    assert g.node_features['node_feat'] is None
    assert g.node_features['node_emb'] is None
    assert g.node_attributes[0]['node_attr'] is None

    g.node_features['zero'] = torch.zeros(10)
    g.add_nodes(9)
    assert g.get_node_num() == 19
    assert torch.all(torch.eq(g.node_features['zero'][10:], torch.zeros(9)))


def test_set_node_features():
    g = GraphData()
    g.add_nodes(10)
    # Test adding node features at whole graph level.
    try:
        g.node_features['node_feat'] = torch.randn((9, 10))
        fail_here()
    except AssertionError:
        pass

    g.node_features['node_feat'] = torch.randn((10, 10))
    g.node_features['zero'] = torch.zeros(10)
    g.node_features['idx'] = torch.tensor(list(range(10)), dtype=torch.long)

    for i in range(g.get_node_num()):
        assert g.node_features['idx'][i] == i
        assert g.node_features['node_feat'][i].shape == torch.Size([10])

    # Test modifying node features partially
    g.node_features['zero'][:5] = torch.ones(5)
    for i in range(5):
        assert g.node_features['zero'][i] == 1

    # Test computational graph consistency when modifying node features both partially and completely
    embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=10)
    optimizer = torch.optim.Adam(params=embedding_layer.parameters())
    target = torch.ones(10)
    init_loss = - torch.nn.functional.cosine_similarity(embedding_layer(g.node_features['idx']).sum(dim=-1), target,
                                                        dim=0)
    for i in range(100):
        g.node_features['node_emb'] = embedding_layer(g.node_features['idx'])
        node_emb_sum = g.node_features['node_emb'].sum(dim=-1)
        loss = - torch.nn.functional.cosine_similarity(node_emb_sum, target, dim=0)
        # print('Epoch {}: loss = {}.'.format(i + 1, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Initial loss = {}. Final loss = {}'.format(init_loss, loss))
    assert init_loss > loss


def test_add_edges():
    g = GraphData()
    g.add_nodes(10)

    # Test wrong cases where one of the `src` and `tgt` is empty.
    try:
        g.add_edges([], [0, 1, 2])
        fail_here()
    except ValueError:
        pass

    try:
        g.add_edges([0, 1, 2], [])
        fail_here()
    except ValueError:
        pass

    # Test `src` and `tgt` size mismatch situation.
    try:
        g.add_edges([0, 1, 2], [1, 2])
        fail_here()
    except ValueError:
        pass

    # Test list autocomplete
    g.add_edges([0], [1, 2])
    g.add_edges([3, 4], [5])
    assert g.get_all_edges() == [(0, 1), (0, 2), (3, 5), (4, 5)]

    # Test normal add_edges
    g = GraphData()
    g.add_nodes(10)
    g.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    assert g.get_all_edges() == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    # Test adding duplicate edges
    with pytest.warns(Warning):
        g.add_edges([0, 1], [1, 2])


def test_edge_ids():
    g = GraphData()
    g.add_nodes(10)
    for i in range(10):
        g.add_edge(src=i, tgt=(i + 1) % 10)
    # all edges are in the graph
    assert g.edge_ids([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]) == [0, 1, 2, 3, 4]
    # some edges are missing
    try:
        g.edge_ids([0, 1, 2], [1, 2, 4])
        fail_here()
    except EdgeNotFoundException:
        pass
    # broadcasting
    g.add_edges(0, [2, 3, 4, 5, 6])
    assert g.edge_ids(0, [1, 2, 3, 4, 5, 6]) == [0, 10, 11, 12, 13, 14]


def test_edge_features():
    g = GraphData()
    g.add_nodes(10)
    for i in range(10):
        g.add_edge(src=i, tgt=(i + 1) % 10)
    g.edge_features['idx'] = torch.tensor(list(range(10)), dtype=torch.long)
    # Test computational graph consistency when modifying node features both partially and completely
    embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=10)
    optimizer = torch.optim.Adam(params=embedding_layer.parameters())
    target = torch.ones(10)
    init_loss = - torch.nn.functional.cosine_similarity(embedding_layer(g.edge_features['idx']).sum(dim=-1), target,
                                                        dim=0)
    for i in range(100):
        g.edge_features['edge_emb'] = embedding_layer(g.edge_features['idx'])
        node_emb_sum = g.edge_features['edge_emb'].sum(dim=-1)
        loss = - torch.nn.functional.cosine_similarity(node_emb_sum, target, dim=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Initial loss = {}. Final loss = {}'.format(init_loss, loss))
    assert init_loss > loss


def test_conversion_dgl():
    g = GraphData()
    g.add_nodes(10)
    for i in range(10):
        g.add_edge(src=i, tgt=(i + 1) % 10)
    g.node_features['node_feat'] = torch.randn((10, 10))
    g.node_features['zero'] = torch.zeros(10)
    g.node_features['idx'] = torch.tensor(list(range(10)), dtype=torch.long)
    g.edge_features['edge_feat'] = torch.randn((10, 10))
    g.edge_features['idx'] = torch.tensor(list(range(10)), dtype=torch.long)
    # Test to_dgl
    dgl_g = g.to_dgl()
    for node_feat_name in g.get_node_feature_names():
        if g.node_features[node_feat_name] is None:
            assert node_feat_name not in dgl_g.ndata.keys()
        else:
            assert torch.all(torch.eq(dgl_g.ndata[node_feat_name], g.node_features[node_feat_name]))
    for edge_feat_name in g.get_edge_feature_names():
        if g.edge_features[edge_feat_name] is None:
            assert edge_feat_name not in dgl_g.edata.keys()
        else:
            assert torch.all(torch.eq(dgl_g.edata[edge_feat_name], g.edge_features[edge_feat_name]))
    assert g.get_node_num() == dgl_g.number_of_nodes()
    src, tgt = dgl_g.all_edges()
    dgl_g_edges = []
    for i in range(src.shape[0]):
        dgl_g_edges.append((int(src[i]), int(tgt[i])))
    assert g.get_all_edges() == dgl_g_edges
    # Test from_dgl
    g1 = from_dgl(dgl_g)
    for node_feat_name in g.get_node_feature_names():
        try:
            assert torch.all(torch.eq(g1.node_features[node_feat_name], g.node_features[node_feat_name]))
        except TypeError:
            assert g1.node_features[node_feat_name] == g.node_features[node_feat_name]
    for edge_feat_name in g.get_edge_feature_names():
        try:
            assert torch.all(torch.eq(g1.edge_features[edge_feat_name], g.edge_features[edge_feat_name]))
        except TypeError:
            assert g1.edge_features[edge_feat_name] == g.edge_features[edge_feat_name]
    assert g1.get_node_num() == g.get_node_num()
    assert g1.get_all_edges() == g.get_all_edges()


def test_batch():
    g_list = []
    batched_edges = []
    graph_edges_list = []
    for i in range(5):
        g = GraphData()
        g.add_nodes(10)
        for j in range(10):
            g.add_edge(src=j, tgt=(j + 1) % 10)
            batched_edges.append((i * 10 + j, i * 10 + ((j + 1) % 10)))
        g.node_features['idx'] = torch.ones(10) * i
        g.edge_features['idx'] = torch.ones(10) * i
        graph_edges_list.append(g.get_all_edges())
        g_list.append(g)

    # Test to_batch
    batch = to_batch(g_list)

    target_batch_idx = []
    for i in range(5):
        for j in range(10):
            target_batch_idx.append(i)

    assert batch.batch == target_batch_idx
    assert batch.get_node_num() == 50
    assert batch.get_all_edges() == batched_edges

    # Test from_batch
    graph_list = from_batch(batch)

    for i in range(len(graph_list)):
        g = graph_list[i]
        assert g.get_all_edges() == graph_edges_list[i]
        assert g.get_node_num() == 10
        assert torch.all(torch.eq(g.node_features['idx'], torch.ones(10) * i))
        assert torch.all(torch.eq(g.edge_features['idx'], torch.ones(10) * i))

    # Test graph with 0 edges
    gl = [GraphData() for _ in range(5)]
    for i in range(5):
        gl[i].add_nodes(1)
    b = to_batch(gl)
    gll = from_batch(b)
    for i in range(5):
        print(gll[i].get_edge_num())
