.. _guide-gat:

Graph Attention Networks
===========


The Graph Attention Network (`GAT <https://arxiv.org/abs/1710.10903>`__) aims to learn edge weights for the input binary adjacency matrix by introducing
the multi-head attention mechanism to the GNN architecture when performing message passing.
We provide high level APIs to users to easily define a multi-layer GAT model. Besides, we support both
regular GAT and bidirectional versions including `GAT-BiSep <https://arxiv.org/abs/1808.07624>`__
and `GAT-BiFuse <https://arxiv.org/abs/1908.04942>`__.
Below is an example to call the GAT API.

.. code-block:: python

    from graph4nlp.pytorch.modules.graph_embedding import GAT

    model = GAT(3, # num of layers
            32, # input size
            16, # hidden size
            2, # output size
            [8, 8, 2], # heads
            direction_option='bi_fuse',
            feat_drop=0.1,
            attn_drop=0.1,
            negative_slope=0.5,
            residual=False)
