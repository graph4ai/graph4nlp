.. _ie-graph-construction:

IE Graph Construction
=====================
The information extraction graph (IE Graph) aims to extract the structural information
to represent the high-level information among natural sentences, e.g., text-based documents.
We divide this process into two three basic steps: 1) coreference resolution,
2) constructing IE relations, and 3) graph construction.

Coreference resolution is the basic procedure for information extraction task which
aims to find expressions that refer to the same entities in the text sequence.
For example, the noun "James" and pronouns "He" may refer to the same object (person)
in sentence "James is on shop. He buys eggs." In this step, all pronouns in the raw paragraph
will be replaced with the corresponding nouns.

To construct an IE graph, the second step is to extract the triples from the processed paragraphs,
which could be completed by leveraging some well-known information extraction systems (i.e. OpenIE).
After this step, we can obtain a list of triples, and a triple can be denoted as :math:`(n_i, r_{i, j}, n_j)`.

For example, we can construct the IE graph given a raw textual input:

.. code:: python

    from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
    from stanfordcorenlp import StanfordCoreNLP

    raw_data = ('James is on shop. He buys eggs.')

    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)

    props_coref = {
                    'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
                    "tokenize.options":
                        "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }

    props_openie = {
        'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
        "tokenize.options":
            "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
        "tokenize.whitespace": False,
        'ssplit.isOneSentence': False,
        'outputFormat': 'json',
        "openie.triple.strict": "true"
    }

    processor_args = [props_coref, props_openie]

    graphdata = IEBasedGraphConstruction.topology(raw_data, nlp_parser,
                                                  processor_args=processor_args,
                                                  merge_strategy=None,
                                                  edge_strategy=None)

Merge strategy
----------
Since the ie graph is only constructed for sentences individually, we provide options to construct one graph
for the paragraph consisting of multiple sentences. Currently, we support the following options:

1. ``None``. Do not add additional nodes and edges and the original subgraphs may not be connected.
2. ``global``. It means all subjects in extracted triples are connected by a "GLOBAL_NODE" using a "global" edge.
3. ``user_define``. We suggest users to define their merge strategy by overriding the ``_graph_connect``.

.. code:: python

    from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction

    class NewIEBasedGraphConstruction(IEBasedGraphConstruction):
        def _graph_connect(cls, triple_list, merge_strategy=None):
            ...

Edge Strategy
----------
There are various dependency relations for ie graph nodes. Currently, we support the following options:

1. ``None``. It means we will not add additional edges.
2. ``as_node``. We will view each edge as a graph node and construct the bipartite graph. For example, if there is an edge whose type is :math:`k` between node :math:`i` and node :math:`j`, we will insert a node :math:`k` into the graph and link node :math:`(i, k)` and :math:`(k, j)`.
