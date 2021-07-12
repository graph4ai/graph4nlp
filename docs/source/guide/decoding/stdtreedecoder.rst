.. _std-tree-decoder:

Chapter 5.2 Standard Tree Decoder
=================================

The output of many NLP applications (i.e., semantic parsing, code generation, and math word problem) contain structural information. For example, the output in math word problem is a mathematical equation, which can be expressed naturally by the data structure of the tree. To model these kinds of outputs, tree decoders are widely adopted. Tree decoders can be divided into two main parts: ``DFS`` (depth-first search) based tree decoder, and ``BFS`` (breadth-first search) based tree decoder. We mainly implement ``BFS`` based tree decoder here. Similar to ``StdRNNDecoder``, ``StdTreeDecoder`` also employ ``copy`` and ``separate attention`` mechanism to enhance the overall ``Graph2Tree`` model.

.. code:: python

    import torch
    import torch.nn as nn
    from graph4nlp.pytorch.modules.config import get_basic_args
    from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config
    
    from graph4nlp.pytorch.datasets.jobs import JobsDatasetForTree
    from graph4nlp.pytorch.modules.utils.tree_utils import Vocab
    from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder
    
    # get your vocab_model, batch_graph, and tgt_tree_batch
    dec_word_emb = nn.Embedding(out_vocab.embeddings.shape[0],
                                 out_vocab.embeddings.shape[1],
                                 padding_idx=0,
                                 _weight=torch.from_numpy(out_vocab.embeddings).float())
    
    decoder = StdTreeDecoder(attn_type="uniform", embeddings=dec_word_emb, enc_hidden_size=300,
                             dec_emb_size=out_vocab.embedding_dims, dec_hidden_size=300,
                             output_size=out_vocab.vocab_size,
                             criterion=nn.NLLLoss(ignore_index=0, reduction='none'),
                             teacher_force_ratio=1.0, use_copy=False, max_dec_seq_length=50,
                             max_dec_tree_depth=5, tgt_vocab=out_vocab)
    
    predicted = decoder(batch_graph=batch_graph, tgt_tree_batch=tgt_tree_batch)