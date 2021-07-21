.. _guide-dynamic_graph_construction:

Dynamic Graph Construction
===========

Unlike static graph construction which is performed during preprocessing,
dynamic graph construction operates by jointly learning the graph structure
and graph representation on the fly. The ultimate goal is to learn the
optimized graph structures and representations with respect to certain
downstream prediction task.
As shown in the figure below, given a set of data points which can stand for
various NLP elements such as words, sentences and documents, we first apply
graph similarity metric learning which aims to capture the pair-wise
node similarity and returns a fully-connected weighted graph.
Then, we can optionally apply graph sparsification to obtain a sparse graph.
When the initial graph topology is available, we can choose to combine the
initial graph topology and the implicit learned graph topology to obtain a
better graph topology for the downstream task.

.. image:: ../imgs/dynamic_graph_overall.pdf



DynamicGraphConstructionBase
-----------------

Before we introduce the two built-in dynamic graph construction classes, let's
first talk about ``DynamicGraphConstructionBase`` which is the base class
for dynamic graph construction. This base class implements several important
components shared by various dynamic graph construction approaches. We will
introduce each of the components defined in the base class next.


The ``embedding`` method aims to compute initial node embeddings which will be
later used for dynamic graph construction. This method calls the ``EmbeddingConstruction``
instance which is initialized in ``GraphConstructionBase`` where ``GraphConstructionBase``
is the base class of all the graph construction classes including both static and dynamic ones.



The ``compute_similarity_metric`` method aims to compute pair-wise node similarity in the node embedding
space where the node embeddings are created by the above ``embedding`` method. The output of this method
is a weighted adjacency matrix which is corresponding to a fully-connected graph.
Various similarity metric functions such as `weighted_cosine`, `attention` and `rbf_kernel` are supported.
Below is the implementation of this method.

.. code-block:: python

    def compute_similarity_metric(self, node_emb, node_mask=None):
        if self.sim_metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                node_vec_t = torch.relu(self.linear_sims[_](node_emb))
                attention += torch.matmul(node_vec_t, node_vec_t.transpose(-1, -2))

            attention /= len(self.linear_sims)
        elif self.sim_metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight.unsqueeze(1)
            if len(node_emb.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            node_vec_t = node_emb.unsqueeze(0) * expand_weight_tensor
            node_vec_norm = F.normalize(node_vec_t, p=2, dim=-1)
            attention = torch.matmul(node_vec_norm, node_vec_norm.transpose(-1, -2)).mean(0)
        elif self.sim_metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](node_emb)
                a_input2 = self.linear_sims2[_](node_emb)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
        elif self.sim_metric_type == 'rbf_kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self._compute_distance_matrix(node_emb, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))
        elif self.sim_metric_type == 'cosine':
            node_vec_norm = node_emb.div(torch.norm(node_emb, p=2, dim=-1, keepdim=True))
            attention = torch.mm(node_vec_norm, node_vec_norm.transpose(-1, -2)).detach()

        if node_mask is not None:
            if torch.__version__ < '1.3.0':
                attention = attention.masked_fill_(~(node_mask == 1.), self.mask_off_val)
            else:
                attention = attention.masked_fill_(~node_mask.bool(), self.mask_off_val)

        return attention


The ``sparsify_graph`` method aims to obtain a sparse graph from the above fully-connected graph.
Various graph sparsification options such as `kNN sparsification` and `epsilon-neighborhood sparsification`
are supported. Below is the implementation of this method.

.. code-block:: python

    def sparsify_graph(self, adj):
        if self.epsilon_neigh is not None:
            adj = self._build_epsilon_neighbourhood(adj, self.epsilon_neigh)

        if self.top_k_neigh is not None:
            adj = self._build_knn_neighbourhood(adj, self.top_k_neigh)

        return adj


The ``compute_graph_regularization`` method aims to compute regularization terms for the learned graph topology.
Various graph regularization losses such as `smoothness`, `connectivity` and `sparsity` are supported.
Below is the implementation of this method.

.. code-block:: python

    def compute_graph_regularization(self, adj, node_feat):
        graph_reg = 0
        if not self.smoothness_ratio in (0, None):
            for i in range(adj.shape[0]):
                L = torch.diagflat(torch.sum(adj[i], -1)) - adj[i]
                graph_reg += self.smoothness_ratio * torch.trace(torch.mm(node_feat[i].transpose(-1, -2), torch.mm(L, node_feat[i]))) / int(np.prod(adj.shape))

        if not self.connectivity_ratio in (0, None):
            ones_vec = torch.ones(adj.shape[:-1]).to(adj.device)
            graph_reg += -self.connectivity_ratio * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).sum() / adj.shape[0] / adj.shape[-1]

        if not self.sparsity_ratio in (0, None):
            graph_reg += self.sparsity_ratio * torch.sum(torch.pow(adj, 2)) / int(np.prod(adj.shape))

        return graph_reg




Node Embedding Based Dynamic Graph Construction
-----------------

For node embedding based dynamic graph construction, we aim to learn the graph structure from a set of node embeddings.
The ``NodeEmbeddingBasedGraphConstruction`` class inherits the ``DynamicGraphConstructionBase`` base class which implements
several aforementioned important components (e.g., ``compute_similarity_metric``, ``sparsify_graph``).
The ``topology`` method in ``NodeEmbeddingBasedGraphConstruction`` implements the logic of learning a graph topology from
initial node embeddings, as shown below:

.. code-block:: python

    def topology(self, graph):
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



Node Embedding Based Refined Dynamic Graph Construction
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
