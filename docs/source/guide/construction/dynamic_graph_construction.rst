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
Below is an example to call the API.

.. code-block:: python


    from graph4nlp.pytorch.modules.graph_construction import NodeEmbeddingBasedGraphConstruction

    embedding_style = {'single_token_item': True if config['graph_type'] != 'ie' else False,
                    'emb_strategy': config.get('emb_strategy', 'w2v_bilstm'),
                    'num_rnn_layers': 1,
                    'bert_model_name': config.get('bert_model_name', 'bert-base-uncased'),
                    'bert_lower_case': True
                   }
    graph_learner = NodeEmbeddingBasedGraphConstruction(
                                        word_vocab, # word vocab instance
                                        embedding_style, # dict specifying the embedding style
                                        sim_metric_type='weighted_cosine', # similarity metric learning type
                                        num_heads=2, # num of similarity metric heads
                                        epsilon_neigh=0.6, # threshold for epsilon-neighborhood sparsification
                                        smoothness_ratio=None, # ratio for graph regularization smothness
                                        connectivity_ratio=None, # ratio for graph regularization connectivity
                                        sparsity_ratio=None, # ratio for graph regularization sparsity
                                        input_size=32,
                                        hidden_size=32,
                                        fix_word_emb=True,
                                        fix_bert_emb=True,
                                        word_dropout=0.4,
                                        rnn_dropout=0.4)



Node Embedding Based Refined Graph
-----------------
Unlike the node embedding based metric learning, node embedding based refined graph metric
learning in addition utilizes the intrinsic graph structure which potentially still carries
rich and useful information regarding the optimal graph structure for the downstream task.
It basically computes a linear combination of the normalized graph Laplacian of the intrinsic
graph and the normalized adjacency matrix of the learned implicit graph.
Below is an example to call the API.

.. code-block:: python

    from graph4nlp.pytorch.modules.graph_construction import NodeEmbeddingBasedRefinedGraphConstruction

    embedding_style = {'single_token_item': True if config['graph_type'] != 'ie' else False,
                    'emb_strategy': config.get('emb_strategy', 'w2v_bilstm'),
                    'num_rnn_layers': 1,
                    'bert_model_name': config.get('bert_model_name', 'bert-base-uncased'),
                    'bert_lower_case': True
                   }
    graph_learner = NodeEmbeddingBasedRefinedGraphConstruction(
                                        word_vocab,
                                        embedding_style,
                                        0.2, # ratio for combining the initial adjacency matrix
                                        sim_metric_type='weighted_cosine',
                                        num_heads=2,
                                        epsilon_neigh=0.6,
                                        smoothness_ratio=None,
                                        connectivity_ratio=None,
                                        sparsity_ratio=None,
                                        input_size=32,
                                        hidden_size=32,
                                        fix_word_emb=True,
                                        fix_bert_emb=True,
                                        word_dropout=0.4,
                                        rnn_dropout=0.4)
