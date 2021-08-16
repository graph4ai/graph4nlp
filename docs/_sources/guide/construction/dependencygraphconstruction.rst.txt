.. _dependency-graph-construction:

Dependency Graph Construction
=====================
The dependency graph is widely used to capture the dependency relations between different objects in the given sentences.
Formally, given a paragraph, one can obtain the dependency parsing tree (e.g., syntactic dependency tree or semantic dependency parsing tree) by using various NLP parsing tools (e.g., Stanford CoreNLP).
Then one may extract the dependency relations from the dependency parsing tree and convert them into a dependency graph.

More concretely, we devide the process into several steps:

1) Parsing. It will parse the input paragraph into list of sentences. Then for each sentence, we will parse the dependency relations.
2) Sub-graph construction. We will construct subgraph for each sentence.
3) Graph merging. We will merge sub-graphs into one big graph.

.. code:: python

    @classmethod
    def topology(cls, raw_text_data, nlp_processor, processor_args, merge_strategy, edge_strategy, sequential_link=True,
                 verbase=0):
        # 1) Parsing
        parsed_results = cls.parsing(raw_text_data=raw_text_data, nlp_processor=nlp_processor,
                                     processor_args=processor_args)

        # 2) Sub-graphs construction.
        sub_graphs = []
        for sent_id, parsed_sent in enumerate(parsed_results):
            graph = cls._construct_static_graph(parsed_sent, edge_strategy=edge_strategy,
                                                sequential_link=sequential_link)
            sub_graphs.append(graph)
        # 3) Graph merging.
        joint_graph = cls._graph_connect(sub_graphs, merge_strategy)
        return joint_graph

How to use
--------------
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

The specific details
--------------------
Parsing
^^^^^^^^^^^^^^
The parsing function first parses the input paragraph into list of sentences. Then for each sentence, we will parse it into dict containing:

1) ``node_num`` indicating the number of nodes.
2) ``node_content``. It is a list of dicts. Each dict is a node.
3) ``graph_content``. It is a list of dicts. Each dict is a dependency relation between the source and target nodes.

Sub-graph construction
^^^^^^^^^^^^^^
In this step, we constrcut sub-graph given parsed sentence results in the previous step controlled by ``edge_strategy`` and ``sequential_link``.

.. code:: python
    @classmethod
    def _construct_static_graph(cls, parsed_object, edge_strategy=None, sequential_link=True):
        ...

We first add the nodes to the graph.

.. code:: python

    ret_graph = GraphData()
    node_objects = parsed_object["node_content"]
    for node in node_objects:
        ret_graph.node_attributes[node['id']]['type'] = 0
        ret_graph.node_attributes[node['id']]['token'] = node['token']
        ret_graph.node_attributes[node['id']]['position_id'] = node['position_id']
        ret_graph.node_attributes[node['id']]['sentence_id'] = node['sentence_id']
        ret_graph.node_attributes[node['id']]['head'] = False
        ret_graph.node_attributes[node['id']]['tail'] = False

Then we will add edges according to the dependency relations. There are various dependency relations for dependency graph nodes. According to the need for down-tasks, we provide several options for: 1. homogeneous graph, 2. heterogeneous graph. Specifically, for heterogeneous graphs, we support not only various graph edge types but also support bipartite graphs, which regarding the edges as special nodes:

1. ``homogeneous``. It means we will drop the edge type information and only preserve the connectivity information.

2. ``heterogeneous``. It means we will preserve the edge type information in the final ``GraphData``. Note that they are stored in the ``edge_attributes`` with ``token`` key.

3. ``as_node``. We will view each edge as a graph node and construct the bipartite graph. For example, if there is an edge whose type is :math:`k` between node :math:`i` and node :math:`j`, we will insert a node :math:`k` into the graph and link node :math:`(i, k)` and :math:`(k, j)`.


.. code:: python

    for dep_info in parsed_object["graph_content"]:
        if edge_strategy is None or edge_strategy == "homogeneous":
            ret_graph.add_edge(dep_info["src"], dep_info['tgt']) # Node edge type, only connectivity information.
        elif edge_strategy == "heterogeneous":
            ret_graph.add_edge(dep_info["src"], dep_info['tgt'])
            edge_idx = ret_graph.edge_ids(dep_info["src"], dep_info['tgt'])[0]
            ret_graph.edge_attributes[edge_idx]["token"] = dep_info["edge_type"] # The node types are stored.
        elif edge_strategy == "as_node":
            # insert a node
            node_idx = ret_graph.get_node_num()
            ret_graph.add_nodes(1)
            ret_graph.node_attributes[node_idx]['type'] = 3  # 3 for edge node
            ret_graph.node_attributes[node_idx]['token'] = dep_info['edge_type']
            ret_graph.node_attributes[node_idx]['position_id'] = None
            ret_graph.node_attributes[node_idx]['head'] = False
            ret_graph.node_attributes[node_idx]['tail'] = False
            # add edge infos
            ret_graph.add_edge(dep_info['src'], node_idx)
            ret_graph.add_edge(node_idx, dep_info['tgt'])
        else:
            raise NotImplementedError()

In addition, the sequential relation encodes the adjacent relation of the elements in the original paragraph.
Specifically, for dependency graph constructing, we define the sequential relation set :math:`\mathcal{R}_{seq} \subseteq \mathcal{V} \times \mathcal{V}`, where :math:`\mathcal{V}` is the basic element (i.e., word) set. For each sequential relation :math:`(w_i, w_{i+1}) \in \mathcal{R}_{seq}`, it means :math:`w_i` is adjacent to :math:`w_{i+1}` in the given paragraph.

.. code:: python
    sequential_list = [i for i in range(node_num)]

    if sequential_link and len(sequential_list) > 1:
        for st, ed in zip(sequential_list[:-1], sequential_list[1:]):
            try:
                ret_graph.edge_ids(st, ed)
            except:
                ret_graph.add_edge(st, ed)
            try:
                ret_graph.edge_ids(ed, st)
            except:
                ret_graph.add_edge(ed, st)
    return ret_graph

Users can set ``sequential_link`` to ``True`` to enable this feature.


Graph merging
^^^^^^^^^^^^^^
Since the dependency graph is only constructed for sentences individually, we provide options to construct one graph
for the paragraph consisting of multiple sentences. Currently, we support the following options:

1. ``tailhead``. It means we will link the tail node of :math:`{i-1}^{th}` sentence's graph with the head node of :math:`i^{th}` sentence's graph.
2. ``user_define``. We suggest users to define their merge strategy by overriding the ``_graph_connect`` as follows:

.. code:: python

    from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction

    class NewDependencyGraphConstruction(DependencyBasedGraphConstruction):
        def _graph_connect(cls, nx_graph_list, merge_strategy=None):
            ...
