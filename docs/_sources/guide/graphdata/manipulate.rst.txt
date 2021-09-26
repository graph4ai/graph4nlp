.. _guide-manipulate:

Manipulating GraphData
===========

After constructing a GraphData instance and adding several nodes and edges to it, the next job is to attach some useful
information to it for further processing. In ``GraphData``, there are two types of information related to the nodes and
edges, namely `features` and `attributes`.

Features
---------

Features are PyTorch tensors designated for each node or edge. To access features, users may call
``GraphData.nodes[node_index].features`` or ``GraphData.edges[index].features``.
These methods return the features of the specified nodes or edges in a dictionary representation, where the keys are the
names of the feature and the values are the corresponding tensors.
Alternatively, if the user wants to access features from the whole-graph level, the ``GraphData.node_features`` and
``GraphData.edge_features`` interfaces do the exact job.

.. code::

    g = GraphData()
    g.add_nodes(10)

    # Note that the first dimension of features represent the number of instances(nodes/edges).
    # Any manipulation to the features should keep the match between the number of instances and the dimension size
    # An invalid example
    g.node_features['node_feat'] = torch.randn((9, 10))
    >>> raise SizeMisatchError

    g.node_features['node_feat'] = torch.randn((10, 10))
    g.node_features['zero'] = torch.zeros(10)
    g.node_features['idx'] = torch.tensor(list(range(10)), dtype=torch.long)
    g.node_features
    >>> {'node_feat': tensor([[-2.2053, -0.9236, -0.4437, -0.7142,  1.5309, -1.5863,  0.6002, -0.6847,
          1.3772,  0.1066],
        [ 0.8875,  1.7674, -0.0354, -0.7681, -2.6256, -1.3399, -2.3798, -0.7418,
          1.2901,  0.6641],
        [-1.5530,  0.9147,  0.0618, -0.0879,  1.0005,  1.2638, -1.4481,  1.2975,
         -0.0304,  0.8707],
        [-0.3448, -0.7484, -1.0194, -0.5096, -0.2596,  0.1056,  1.1560,  0.3463,
         -0.1986,  0.9243],
        [-0.3555, -0.7062, -1.0459,  0.1305, -0.1338,  1.2952,  1.2923, -0.5740,
         -0.5492, -0.2497],
        [-0.7125,  1.2456, -0.2136,  0.8562,  1.8037, -0.0379, -1.6863,  1.2693,
         -0.1980, -0.3153],
        [ 0.4099, -0.8295,  0.6984,  0.4125, -0.8396,  1.8205, -1.1458, -0.0837,
         -0.2388,  0.0552],
        [-1.4068, -1.9334, -0.0367, -1.3297,  1.0705, -0.5606, -0.0458,  0.1358,
          1.3042, -0.8282],
        [ 0.7764,  0.1442,  1.6043,  0.1052,  1.4648, -2.1791,  0.6740,  0.2858,
          0.0482,  0.9058],
        [-1.5054,  0.8992,  0.0893, -1.2325,  0.8888, -1.2222,  2.0569,  0.0218,
          1.5519, -0.8234]]),
        'node_emb': None,
        'zero': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        'idx': tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

Note that there are some reserved keys for features, which are initialized to be ``None``. In node features the reserved keys
are `node_feat` and `node_emb`. In edge features the reserved keys are `edge_feat`, `edge_emb` and `edge_weight`.
Users are encouraged to use these keys as common feature names.
This means that the feature dictionary of an empty graph will have these items with the corresponding values being ``None``.

Another thing to notice is that when adding new nodes or edges to a graph whose features are already set(the value is not
None), zero padding will be performed on the newly added instances.

.. code::

    g.add_nodes(1)
    g.node_features     # Zero padding is performed
    >>> {'node_feat': tensor([[-2.2053, -0.9236, -0.4437, -0.7142,  1.5309, -1.5863,  0.6002, -0.6847,
          1.3772,  0.1066],
        [ 0.8875,  1.7674, -0.0354, -0.7681, -2.6256, -1.3399, -2.3798, -0.7418,
          1.2901,  0.6641],
        [-1.5530,  0.9147,  0.0618, -0.0879,  1.0005,  1.2638, -1.4481,  1.2975,
         -0.0304,  0.8707],
        [-0.3448, -0.7484, -1.0194, -0.5096, -0.2596,  0.1056,  1.1560,  0.3463,
         -0.1986,  0.9243],
        [-0.3555, -0.7062, -1.0459,  0.1305, -0.1338,  1.2952,  1.2923, -0.5740,
         -0.5492, -0.2497],
        [-0.7125,  1.2456, -0.2136,  0.8562,  1.8037, -0.0379, -1.6863,  1.2693,
         -0.1980, -0.3153],
        [ 0.4099, -0.8295,  0.6984,  0.4125, -0.8396,  1.8205, -1.1458, -0.0837,
         -0.2388,  0.0552],
        [-1.4068, -1.9334, -0.0367, -1.3297,  1.0705, -0.5606, -0.0458,  0.1358,
          1.3042, -0.8282],
        [ 0.7764,  0.1442,  1.6043,  0.1052,  1.4648, -2.1791,  0.6740,  0.2858,
          0.0482,  0.9058],
        [-1.5054,  0.8992,  0.0893, -1.2325,  0.8888, -1.2222,  2.0569,  0.0218,
          1.5519, -0.8234],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000]]),
        'node_emb': None,
        'zero': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        'idx': tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])}


Attributes
-------------

Another important attached information is `attribute`. Like `features`, `attribute` is related to each node/edge
instance and is basically a list of dictionaries. The list index corresponds to the node/edge index and the dictionary
at each position stands for the corresponding attributes of that instance.
Essentially, `attribute` is designed to make up for the limit of `features` in storing arbitrary objects. The reserved
keys are `node_attr` for node attributes and `edge_attr` for edge attributes.

.. code::

    g = GraphData()
    g.add_nodes(2)  # Add 2 nodes to an empty graph
    g.node_attributes
    >>> [{'node_attr': None}, {'node_attr': None}]
    g.node_attributes[1]['node_attr'] = 'hello'
    g.node_attributes
    >>> [{'node_attr': None}, {'node_attr': 'hello'}]


Features vs. Attributes
----------------

To make it clear, in this subsection we compare the differences between features and attributes in order for users to
better utilize them.

1. Types of storage

``features`` store only the numerical feature objects. In current version these data are PyTorch tensors. The shape of these
tensor data should be consistent with the number of nodes/edges in the graph. Specifically, the first dimension
of the tensor data corresponds to the number of instances. For example, in a graph with 10 nodes and 20 edges, the shape
of any node feature tensor should be [10, *] and [20, *] for any edge feature.

On the other hand, ``attributes`` store arbitrary type of data. The data can be of any type and do not necessarily need
to have a ``shape``.

2. Order of access

Both ``features`` and ``attributes`` have two levels of keys: *names* and *indices*. ``features`` are implemented as a
dictionary where the keys are strings and values are tensors. Therefore, the first level of key is the feature names.
In this way, the second level of keys are just direct access to the corresponding PyTorch tensors.

On the other hand, ``attributes`` are implemented as a list of dictionaries, where the list indices are the node indices.
Therefore, when accessing attributes, users should use the index first.

