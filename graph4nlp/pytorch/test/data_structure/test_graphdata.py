import gc
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import pytest

from ...data.data import GraphData, from_batch, from_dgl, to_batch
from ...data.utils import EdgeNotFoundException, SizeMismatchException


def fail_here():
    raise Exception("The above line of code shouldn't be executed normally")


def test_add_nodes():
    g = GraphData()
    try:
        g.add_nodes(-1)
        fail_here()
    except AssertionError:
        pass

    g.add_nodes(10)
    assert g.get_node_num() == 10  # Test node number

    # Test reserved fields
    assert g.node_features["node_feat"] is None
    assert g.node_features["node_emb"] is None
    assert g.node_attributes[0]["node_attr"] is None

    g.node_features["zero"] = torch.ones(10)
    g.add_nodes(9)
    assert g.get_node_num() == 19
    assert torch.all(torch.eq(g.node_features["zero"][10:], torch.zeros(9)))


def test_set_node_features_cpu():
    g = GraphData()
    g.add_nodes(10)

    # Test adding node features at whole graph level.
    try:
        g.node_features["node_feat"] = torch.randn((9, 10))
        fail_here()
    except SizeMismatchException:
        pass

    g.node_features["node_feat"] = torch.randn((10, 10))
    g.node_features["zero"] = torch.zeros(10)
    g.node_features["idx"] = torch.tensor(list(range(10)), dtype=torch.long)

    for i in range(g.get_node_num()):
        assert g.node_features["idx"][i] == i
        assert g.node_features["node_feat"][i].shape == torch.Size([10])

    # Test modifying node features partially
    g.node_features["zero"][:5] = torch.ones(5)
    for i in range(5):
        assert g.node_features["zero"][i] == 1

    embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=10)
    optimizer = torch.optim.Adam(params=embedding_layer.parameters())
    target = torch.ones(10)
    init_loss = -torch.nn.functional.cosine_similarity(
        embedding_layer(g.node_features["idx"]).sum(dim=-1), target, dim=0
    )
    for _ in range(100):
        g.node_features["node_emb"] = embedding_layer(g.node_features["idx"])
        node_emb_sum = g.node_features["node_emb"].sum(dim=-1)
        loss = -torch.nn.functional.cosine_similarity(node_emb_sum, target, dim=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Initial loss = {}. Final loss = {}".format(init_loss, loss))
    assert init_loss > loss


def test_set_node_features_gpu():
    g = GraphData()
    g.add_nodes(10)
    device = torch.device("cuda:0")
    g.to(device)

    # Test adding node features at whole graph level.
    try:
        g.node_features["node_feat"] = torch.randn((9, 10))
        fail_here()
    except SizeMismatchException:
        pass

    g.node_features["node_feat"] = torch.randn((10, 10))
    g.node_features["zero"] = torch.zeros(10)
    g.node_features["idx"] = torch.tensor(list(range(10)), dtype=torch.long)

    assert g.node_features["node_feat"].device == device

    for i in range(g.get_node_num()):
        assert g.node_features["idx"][i] == i
        assert g.node_features["node_feat"][i].shape == torch.Size([10])

    # Test modifying node features partially
    g.node_features["zero"][:5] = torch.ones(5)
    for i in range(5):
        assert g.node_features["zero"][i] == 1

    embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=10).to(device)
    optimizer = torch.optim.Adam(params=embedding_layer.parameters())
    target = torch.ones(10).to(device)
    init_loss = -torch.nn.functional.cosine_similarity(
        embedding_layer(g.node_features["idx"]).sum(dim=-1), target, dim=0
    )
    for _ in range(100):
        g.node_features["node_emb"] = embedding_layer(g.node_features["idx"])
        node_emb_sum = g.node_features["node_emb"].sum(dim=-1)
        loss = -torch.nn.functional.cosine_similarity(node_emb_sum, target, dim=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Initial loss = {}. Final loss = {}".format(init_loss, loss))
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
    g.edge_features["idx"] = torch.tensor(list(range(10)), dtype=torch.long)
    embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=10)
    optimizer = torch.optim.Adam(params=embedding_layer.parameters())
    target = torch.ones(10)
    init_loss = -torch.nn.functional.cosine_similarity(
        embedding_layer(g.edge_features["idx"]).sum(dim=-1), target, dim=0
    )
    for _ in range(100):
        g.edge_features["edge_emb"] = embedding_layer(g.edge_features["idx"])
        node_emb_sum = g.edge_features["edge_emb"].sum(dim=-1)
        loss = -torch.nn.functional.cosine_similarity(node_emb_sum, target, dim=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Initial loss = {}. Final loss = {}".format(init_loss, loss))
    assert init_loss > loss


def test_scipy_sparse_adj():
    g_list = []
    batched_edges = []
    graph_edges_list = []
    for i in range(5):
        g = GraphData()
        g.add_nodes(10)
        for j in range(10):
            g.add_edge(src=j, tgt=(j + 1) % 10)
            batched_edges.append((i * 10 + j, i * 10 + ((j + 1) % 10)))
        g.node_features["idx"] = torch.ones(10) * i
        g.edge_features["idx"] = torch.ones(10) * i
        graph_edges_list.append(g.get_all_edges())
        g_list.append(g)

    # Test to_batch
    batch = to_batch(g_list)

    adj = batch.sparse_adj(batch_view=True)
    print(adj)


def test_batch_split_features():
    g_list = []
    batched_edges = []
    graph_edges_list = []
    for i in range(5):
        g = GraphData()
        g.add_nodes(10)
        for j in range(10):
            g.add_edge(src=j, tgt=(j + 1) % 10)
            batched_edges.append((i * 10 + j, i * 10 + ((j + 1) % 10)))
        g.node_features["idx"] = torch.ones(10) * i
        g.edge_features["idx"] = torch.ones(10) * i
        graph_edges_list.append(g.get_all_edges())
        g_list.append(g)
    g = GraphData()
    g.add_nodes(11)
    for j in range(11):
        g.add_edge(src=j, tgt=(j + 1) % 11)
    g.node_features["idx"] = torch.ones(11) * 5
    g.edge_features["idx"] = torch.ones(11) * 5
    graph_edges_list.append(g.get_all_edges())
    g_list.append(g)

    # Test to_batch
    batch = to_batch(g_list)
    init_feature = torch.rand(size=(61, 100))
    split_f = batch.split_features(init_feature, "node")
    print(split_f)


def test_conversion_dgl():
    g = GraphData()
    g.add_nodes(10)
    for i in range(10):
        g.add_edge(src=i, tgt=(i + 1) % 10)
    g.node_features["node_feat"] = torch.randn((10, 10))
    g.node_features["zero"] = torch.zeros(10)
    g.node_features["idx"] = torch.tensor(list(range(10)), dtype=torch.long)
    g.edge_features["edge_feat"] = torch.randn((10, 10))
    g.edge_features["idx"] = torch.tensor(list(range(10)), dtype=torch.long)
    # Test to_dgl
    dgl_g = g.to_dgl()
    for node_feat_name in g.node_feature_names():
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
    for node_feat_name in g.node_feature_names():
        try:
            assert torch.all(
                torch.eq(g1.node_features[node_feat_name], g.node_features[node_feat_name])
            )
        except TypeError:
            assert g1.node_features[node_feat_name] == g.node_features[node_feat_name]
    for edge_feat_name in g.get_edge_feature_names():
        try:
            assert torch.all(
                torch.eq(g1.edge_features[edge_feat_name], g.edge_features[edge_feat_name])
            )
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
        g.node_features["idx"] = torch.ones(10) * i
        g.edge_features["idx"] = torch.ones(10) * i
        graph_edges_list.append(g.get_all_edges())
        g_list.append(g)

    # Test to_batch
    batch = to_batch(g_list)

    target_batch_idx = []
    for i in range(5):
        for _ in range(10):
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
        assert torch.all(torch.eq(g.node_features["idx"], torch.ones(10) * i))
        assert torch.all(torch.eq(g.edge_features["idx"], torch.ones(10) * i))

    # Test graph with 0 edges
    gl = [GraphData() for _ in range(5)]
    for i in range(5):
        gl[i].add_nodes(1)
    b = to_batch(gl)
    gll = from_batch(b)
    for i in range(5):
        print(gll[i].get_edge_num())


def test_batch_feature_view():
    # A batch composed of 5 graphs with node num being 10, 20, ..., 50 respectively
    graphs = []
    features = []
    for i in range(5):
        j = i + 1
        g = GraphData()
        g.add_nodes(j * 10)
        feature = torch.ones(size=(g.get_node_num(), 100))
        g.node_features["node_feat"] = feature
        graphs.append(g)
        features.append(feature)

    batch_graph = to_batch(graphs)
    assert torch.all(
        torch.eq(
            batch_graph.batch_node_features["node_feat"],
            torch.nn.utils.rnn.pad_sequence(features, batch_first=True),
        )
    )

    assert batch_graph.split_node_features["node_feat"][0].shape == (10, 100)
    assert batch_graph.split_node_features["node_feat"][1].shape == (20, 100)
    assert batch_graph.split_node_features["node_feat"][2].shape == (30, 100)
    assert batch_graph.split_node_features["node_feat"][3].shape == (40, 100)
    assert batch_graph.split_node_features["node_feat"][4].shape == (50, 100)


def generate_sequential_graphs(bs=5, num_nodes=10):
    start_index = list(range(0, num_nodes - 1, 1))
    end_index = list(range(1, num_nodes, 1))
    graphs = []
    node_features = []
    node_features_2 = []
    edge_features = []
    edge_features_2 = []
    for i in range(bs):
        g = GraphData()
        g.add_nodes(num_nodes)
        g.add_edges(src=start_index, tgt=end_index)
        node_feat = torch.FloatTensor([i + 1] * num_nodes)
        node_feat_2 = torch.FloatTensor([i] * num_nodes)
        g.node_features["node_feat"] = node_feat
        node_features.append(node_feat)
        node_features_2.append(node_feat_2)

        edge_feat = torch.FloatTensor([(i + 1) * 10] * (num_nodes - 1))
        edge_feat_2 = torch.FloatTensor([i * 10] * (num_nodes - 1))
        g.edge_features["edge_feat"] = edge_feat
        edge_features.append(edge_feat)
        edge_features_2.append(edge_feat_2)
        graphs.append(g)
    return edge_features, edge_features_2, graphs, node_features, node_features_2


def test_batch_node_features():
    (
        edge_features,
        edge_features_2,
        graphs,
        node_features,
        node_features_2,
    ) = generate_sequential_graphs(bs=5)

    batch = to_batch(graphs)
    batch_node_features = batch.batch_node_features
    batch_edge_features = batch.batch_edge_features
    assert torch.all(
        torch.eq(
            batch_node_features["node_feat"],
            torch.nn.utils.rnn.pad_sequence(node_features, batch_first=True),
        )
    )
    assert torch.all(
        torch.eq(
            batch_edge_features["edge_feat"],
            torch.nn.utils.rnn.pad_sequence(edge_features, batch_first=True),
        )
    )

    new_batch_edge_features = torch.stack(edge_features_2).unsqueeze(-1)
    batch.batch_edge_features["edge_feat"] = new_batch_edge_features
    graphs = from_batch(batch)
    for i in range(len(graphs)):
        g = graphs[i]
        assert torch.all(torch.eq(g.edge_features["edge_feat"], new_batch_edge_features[i]))


def mem_report():
    """Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported"""

    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation"""
        print("Storage on %s" % (mem_type))
        print("-" * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print("%s\t\t%s\t\t%.2f" % (element_type, size, mem))
        print("-" * LEN)
        print("Total Tensors: %d \tUsed Memory Space: %.2f MBytes" % (total_numel, total_mem))
        print("-" * LEN)

    LEN = 65
    print("=" * LEN)
    objects = gc.get_objects()
    print("%s\t%s\t\t\t%s" % ("Element type", "Size", "Used MEM(MBytes)"))
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    print("=" * LEN)


def test_batch_feat_perf():
    batch_size_list = [5, 10, 20, 40, 60, 100, 150, 200, 400, 600, 800, 1000]
    times = []
    for bs in batch_size_list:
        (
            edge_features,
            edge_features_2,
            graphs,
            node_features,
            node_features_2,
        ) = generate_sequential_graphs(bs=bs)
        batch = to_batch(graphs)
        new_batch_edge_features = torch.stack(edge_features_2).unsqueeze(-1)
        start = time.time()
        batch.batch_edge_features["edge_feat"] = new_batch_edge_features
        time_elapsed = time.time() - start
        times.append(time_elapsed)

    plt.plot(batch_size_list, times)
    plt.title("Time vs batch size")
    plt.show()


def test_batch_feat_perf_nnodes():
    batch_size_list = [5, 10, 20, 40, 60, 100, 150, 200, 400, 600, 800, 1000]
    times = []
    for bs in batch_size_list:
        (
            edge_features,
            edge_features_2,
            graphs,
            node_features,
            node_features_2,
        ) = generate_sequential_graphs(num_nodes=bs)
        batch = to_batch(graphs)
        new_batch_edge_features = torch.stack(edge_features_2).unsqueeze(-1)
        start = time.time()
        batch.batch_edge_features["edge_feat"] = new_batch_edge_features
        time_elapsed = time.time() - start
        times.append(time_elapsed)

    plt.plot(batch_size_list, times)
    plt.title("Time vs #node")
    plt.show()


def test_remove_edges():
    g = GraphData()
    g.add_nodes(10)
    g.add_edges(list(range(0, 9, 1)), list(range(1, 10, 1)))
    g.edge_features["random"] = torch.rand((9, 1024, 1024))
    mem_report()
    g.remove_all_edges()
    mem_report()
