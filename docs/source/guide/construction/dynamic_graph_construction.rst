.. _guide-dynamic_graph_construction:

Dynamic Graph Construction
===========

Unlike static graph construction which is performed during preprocessing,
dynamic graph construction operates by jointly learning the graph structure
and graph representation on the fly. The ultimate goal is to learn the
optimized graph structures and representations with respect to certain
downstream prediction task.
Given a set of data points which can stand for various NLP elements such as words,
sentences and documents, we first apply graph similarity metric learning which aims
to capture the pair-wise node similarity and returns a fully-connected weighted graph.
Then, we can optionally apply graph sparsification to obtain a sparse graph.



Node Embedding Based Graph
-----------------
For node embedding based similarity metric learning, we are given a set of data points
and their vector representations, and apply various metric functions to learn pair-wise
node similarity. The output is a weighted adjacency matrix which is corresponding to a
fully-connected weighted graph. Optional graph sparsification operation might be applied
to obtain a sparse graph. Common similarity metric functions include attention-based and cosine-based functions.


``NodeEmbeddingBasedGraphConstruction`` class inherits the ``DynamicGraphConstructionBase`` base class which performs
several important steps including computing initial node embeddings (see ``embedding`` method),
computing graph similarity metric function (see ``compute_similarity_metric`` method),
sparsifying adjacency matrix (see ``sparsify_graph`` method),
and computing graph regularization loss (see ``compute_graph_regularization`` method).

The ``topology`` method in ``NodeEmbeddingBasedGraphConstruction`` implements the logic of learning a graph topology from
initial node embeddings, as shown below:

.. code-block:: python

    def topology(self, graph):
        """Compute graph topology.

        Parameters
        ----------
        graph : GraphData
            The input graph data.

        Returns
        -------
        GraphData
            The constructed graph.
        """
        node_emb = graph.batch_node_features["node_feat"]
        node_mask = (graph.batch_node_features["token_id"] != Vocab.PAD)

        raw_adj = self.compute_similarity_metric(node_emb, node_mask)
        raw_adj = self.sparsify_graph(raw_adj)
        graph_reg = self.compute_graph_regularization(raw_adj, node_emb)

        if self.sim_metric_type in ('rbf_kernel', 'weighted_cosine'):
            assert raw_adj.min().item() >= 0, 'adjacency matrix must be non-negative!'
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=torch.finfo(torch.float32).eps)
            reverse_adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-2, keepdim=True), min=torch.finfo(torch.float32).eps)
        elif self.sim_metric_type == 'cosine':
            raw_adj = (raw_adj > 0).float()
            adj = normalize_adj(raw_adj)
            reverse_adj = adj
        else:
            adj = torch.softmax(raw_adj, dim=-1)
            reverse_adj = torch.softmax(raw_adj, dim=-2)

        graph = convert_adj_to_graph(graph, adj, reverse_adj, 0)
        graph.graph_attributes['graph_reg'] = graph_reg

        return graph



Node Embedding Based Refined Graph
-----------------
Unlike the node embedding based metric learning, node embedding based refined graph metric
learning in addition utilizes the intrinsic graph structure which potentially still carries
rich and useful information regarding the optimal graph structure for the downstream task.
It basically computes a linear combination of the normalized graph Laplacian of the intrinsic
graph and the normalized adjacency matrix of the learned implicit graph.

``NodeEmbeddingBasedRefinedGraphConstruction`` class also inherits the ``DynamicGraphConstructionBase`` base class.
The ``topology`` method in ``NodeEmbeddingBasedRefinedGraphConstruction`` implements the logic of combining the initial
graph topology and the learned implicit graph topology, as shown below:


.. code-block:: python

    def topology(self, graph, init_norm_adj):
        """Compute graph topology.

        Parameters
        ----------
        graph : GraphData
            The input graph data.
        init_norm_adj : torch.sparse.FloatTensor
            The initial init_norm_adj adjacency matrix.

        Returns
        -------
        GraphData
            The constructed graph.
        """
        node_emb = graph.batch_node_features["node_feat"]
        node_mask = (graph.batch_node_features["token_id"] != Vocab.PAD)

        raw_adj = self.compute_similarity_metric(node_emb, node_mask)
        raw_adj = self.sparsify_graph(raw_adj)
        graph_reg = self.compute_graph_regularization(raw_adj, node_emb)

        if self.sim_metric_type in ('rbf_kernel', 'weighted_cosine'):
            assert raw_adj.min().item() >= 0, 'adjacency matrix must be non-negative!'
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=torch.finfo(torch.float32).eps)
            reverse_adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-2, keepdim=True), min=torch.finfo(torch.float32).eps)
        elif self.sim_metric_type == 'cosine':
            raw_adj = (raw_adj > 0).float()
            adj = normalize_adj(raw_adj)
            reverse_adj = adj
        else:
            adj = torch.softmax(raw_adj, dim=-1)
            reverse_adj = torch.softmax(raw_adj, dim=-2)

        if self.alpha_fusion is not None:
            adj = torch.sparse.FloatTensor.add((1 - self.alpha_fusion) * adj, self.alpha_fusion * init_norm_adj)
            reverse_adj = torch.sparse.FloatTensor.add((1 - self.alpha_fusion) * reverse_adj, self.alpha_fusion * init_norm_adj)

        graph = convert_adj_to_graph(graph, adj, reverse_adj, 0)
        graph.graph_attributes['graph_reg'] = graph_reg

        return graph
