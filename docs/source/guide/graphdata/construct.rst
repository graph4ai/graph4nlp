.. _guide-construct:

Constructing GraphData
===========

This section illustrates the ways to construct GraphData instances.

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
``from_dgl()``, ``from_dense_adj()``, ``from_scipy_sparse_matrix()`` and ``from_graphdata()``. Interested readers may go to the
API reference for more details.

Exporting to other formats
-----------
On the other hand, GraphData also supports converting itself to other formats. Currently ``to_dgl()`` is provided to export
a ``GraphData`` to ``dgl.DGLGraph``.
