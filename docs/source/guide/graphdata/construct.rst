.. _guide-construct:

Constructing GraphData
===========

This section illustrates how to construct GraphData instances.

Constructing from scratch
---------
The most basic way is to construct a GraphData instance from scratch, as the following example shows:

.. code::

    g = GraphData() # Construct an empty graph
    g.add_nodes(10) # Add 10 nodes to this graph sequentially.
    g.get_node_num()
    >>> 10
    g.add_nodes(9)  # Add another 9 nodes. This operation will append the new nodes to existing ones.
    g.get_node_num()
    >>> 19
    g.add_edges([0, 1, 2], [1, 2, 3])   # Add 3 edges, connecting nodes 0~4 into a line.
    g.get_all_edges()
    >>> [(0, 1), (1, 2), (2, 3)]

You may want to access the nodes or edges after adding them. ``GraphData`` provides two interfaces ``GraphData.nodes()``
and ``GraphData.edges()`` to do this job.
However, it may be more illustrative to show the examples with features and attributes attached to nodes and edges, which
makes it more suitable to appear in the next section :ref:`guide-manipulate`.

Importing from other sources
-----------
Another way to construct GraphData is to import from other sources. Currently GraphData supports importing from
``from_dgl()``, ``from_dense_adj()``, ``from_scipy_sparse_matrix()`` and ``from_graphdata()``.
Interested readers may refer to the ``API reference`` for more details.

.. code-block::

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


Exporting to other formats
-----------
On the other hand, GraphData also supports converting itself to other formats. Currently ``to_dgl()`` is provided to export
a ``GraphData`` to ``dgl.DGLGraph``.
