.. _constituency-graph-construction:

Constituency Graph Construction
===============================
The constituency graph is a widely used static graph that is able to capture phrase-based syntactic relations in a sentence. Constituency parsing models the assembly of one or several corresponded words (i.e., phrase level). Thus it provides new insight into the grammatical structure of a sentence.

For example, we can construct the constituency graph given a raw textual input:

.. code:: python

    from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
    from stanfordcorenlp import StanfordCoreNLP
    
    raw_data = "James went to the corner-shop. And bought some eggs."
    
    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)
    
    processor_args = {
        'annotators': 'tokenize,ssplit,pos,parse',
        "tokenize.options":
            "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
        "tokenize.whitespace": True,
        'ssplit.isOneSentence': False,
        'outputFormat': 'json'
    }
    
    graphdata = ConstituencyBasedGraphConstruction.topology(raw_data, nlp_parser, processor_args=processor_args, merge_strategy=None, edge_strategy=None, verbase=True)


Some Options
------------

Sequential Link
```````````````
We provide an option to add no/unidirectional/bidirectional edges between word nodes and nodes connecting two sub-graphs.

1. ``sequential_link`` = ``0``. Do not add sequential links.
2. ``sequential_link`` = ``1``. Add unidirectional links.
3. ``sequential_link`` = ``2``. Add bidirectional links.
4. ``sequential_link`` = ``3``. Do not add sequential links inside each sentence; but add bidirectional links between adjacent sentences.


Prune
`````
The hierarchical structure of constituency graph is complicated, we, therefore, provide some pruning options as follows (``ROOT`` node are pruned by default),

1. ``prune`` = ``0``. No pruning.
2. ``prune`` = ``1``. Prune pos (part-of-speech) nodes.
3. ``prune`` = ``2``. Prune nodes with both in-degree and out-degree of 1.

Merge Strategy
``````````````
Since the constituency graph is only constructed for sentences individually, we provide options to construct one graph
for the paragraph consisting of multiple sentences. Currently, we support the following options:

1. ``tailhead``. It means we will link the tail node of :math:`{i-1}^{th}` sentence's graph with the head node of :math:`i^{th}` sentence's graph.
2. ``user_define``. We suggest users to define their merge strategy by overriding the ``_graph_connect`` as follows:

.. code:: python

    from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction

    class NewConstituencyGraphConstruction(ConstituencyBasedGraphConstruction):
        def _graph_connect(cls, nx_graph_list, merge_strategy=None):
            ...