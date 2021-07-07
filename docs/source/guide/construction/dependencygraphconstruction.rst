.. _dependency-graph-construction:

Chapter 3.1 Dependency Graph Construction
=====================
The dependency graph is widely used to capture the dependency relations between different objects in the given sentences.
Formally, given a paragraph, one can obtain the dependency parsing tree (e.g., syntactic dependency tree or semantic dependency parsing tree) by using various NLP parsing tools (e.g., Stanford CoreNLP).
Then one may extract the dependency relations from the dependency parsing tree and convert them into a dependency graph.

For example, we can construct the dependency graph given a raw textual input:

.. code:: python

    from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
    from stanfordcorenlp import StanfordCoreNLP

    raw_data = "James went to the corner-shop."

    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)

    processor_args = {
        'annotators': 'ssplit,tokenize,depparse',
        "tokenize.options":
            "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
        "tokenize.whitespace": True,
        'ssplit.isOneSentence': True,
        'outputFormat': 'json'
    }

    graphdata = DependencyBasedGraphConstruction.topology(raw_data, nlp_parser, processor_args=processor_args, merge_strategy=None,
                                                          edge_strategy=None)

Merge strategy
----------
Since the dependency graph is only constructed for sentences individually, we provide options to construct one graph
for the paragraph consisting of multiple sentences. Currently, we support the following options:

1. ``tailhead``. It means we will link the tail node of :math:`{i-1}^{th}` sentence's graph with the head node of :math:`i^{th}` sentence's graph.
2. ``user_define``. We suggest users to define their merge strategy by overriding the ``_graph_connect`` as follows:

.. code:: python

    from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction

    class NewDependencyGraphConstruction(DependencyBasedGraphConstruction):
        def _graph_connect(cls, nx_graph_list, merge_strategy=None):
            ...

Edge Strategy
----------
There are various dependency relations for dependency graph nodes. According to the need for down-tasks, we provide several options for: 1. homogeneous graph, 2. heterogeneous graph. Specifically, for heterogeneous graphs, we support not only various graph edge types but also support bipartite graphs, which regarding the edges as special nodes:
1. ``homogeneous``. It means we will drop the edge type information and only preserve the connectivity information.
2. ``heterogeneous``. It means we will preserve the edge type information in the final ``GraphData``. Note that they are stored in the ``edge_attributes`` with ``token`` key.
3. ``as_node``. We will view each edge as a graph node and construct the bipartite graph. For example, if there is an edge whose type is :math:`k` between node :math:`i` and node :math:`j`, we will insert a node :math:`k` into the graph and link node :math:`(i, k)` and :math:`(k, j)`.

Sequential Link
----------
The sequential relation encodes the adjacent relation of the elements in the original paragraph.
Specifically, for dependency graph constructing, we define the sequential relation set :math:`\mathcal{R}_{seq} \subseteq \mathcal{V} \times \mathcal{V}`, where :math:`\mathcal{V}` is the basic element (i.e., word) set. For each sequential relation :math:`(w_i, w_{i+1}) \in \mathcal{R}_{seq}`, it means :math:`w_i` is adjacent to :math:`w_{i+1}` in the given paragraph.

Users can set ``sequential_link`` to ``True`` to enable this feature.
