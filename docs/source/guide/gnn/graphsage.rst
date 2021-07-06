.. _guide-graphsage:

GraphSAGE:
===========


GraphSAGE (`GraphSAGE <https://arxiv.org/pdf/1706.02216.pdf>`__) is a framework for inductive representation learning on large graphs. GraphSAGE is used to generate low-dimensional vector representations for nodes, and is especially useful for graphs that have rich node attribute information. The math operation of GraphSAGE is represented as below:

.. math::
    h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
    \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
    h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
    (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)
    h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})


We provide high level APIs to users to easily define a multi-layer GraphSage model. Besides, we support both
regular GraphSAGE and bidirectional versions including `GraphSAGE-BiSep <https://arxiv.org/abs/1808.07624>`__
and `GraphSAGE BiFuse <https://arxiv.org/abs/1908.04942>`__.
Below is an example to call the GraphSAGE API.

.. code-block:: python

    from graph4nlp.pytorch.modules.graph_embedding import GraphSAGE

    model = GraphSAGE(3, # num of layers
            32, # input size
            16, # hidden size
            2, # output size
            direction_option='bi_fuse',
            feat_drop=0.1,
            bias=True, # whether add a learnable Bia in output
            norm=None,
            activation=None,
            use_edge_weight=False #whether have edge weights)


After define a GraphSAGE model, we can use it to get the node embedding for the input graph. The generated embedding will be automatically stored in the updated graph, as shown in the below example:

.. code-block:: python
 
    updated_Graph = model(input_Graph)



To make the utilization of GraphSAGE more felxbible, we also provide the low-level implementation of GraphSAGE layer. Below is an example to call the GraphSAGE layer API.

.. code-block:: python

    from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGELayer

    layer = GraphSAGELayer(32, # input size
            16, # output size
            direction_option='bi_fuse',
            feat_drop=0.1,
            bias=True, # whether add a learnable Bia in output
            norm=None,
            activation=None)


After define a GraphSAGE layer, we can use it to get the node embedding for the input graph. The generated embedding is the output of this layer, as shown in the below example:
    node_emb = layer(input_graph, 
               feat, # the node feature in tensor format
               edge_weight, # edge weight in tensor format; Only needed when consider the edge weights in message passing
               reverse_edge_weight #whether to use the reversed edge weight)



