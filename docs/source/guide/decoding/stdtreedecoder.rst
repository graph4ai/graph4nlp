.. _std-tree-decoder:

Chapter 5.2 Standard Tree Decoder
=================================

The output of many NLP applications (i.e., semantic parsing, code generation, and math word problem) contain structural information. For example, the output in math word problem is a mathematical equation, which can be expressed naturally by the data structure of the tree. To model these kinds of outputs, tree decoders are widely adopted. Tree decoders can be divided into two main parts: ``DFS`` (depth-first search) based tree decoder, and ``BFS`` (breadth-first search) based tree decoder. We mainly implement ``BFS`` based tree decoder here. Similar to ``StdRNNDecoder``, ``StdTreeDecoder`` also employ ``copy`` and ``separate attention`` mechanism to enhance the overall ``Graph2Tree`` model. For detailed implementation, please refer to :ref:`std-rnn-decoder` or source code.

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

Some detailed descriptions about tree decoding process
------------------------------------------------------
In the BFS-based tree decoding approach, we represent all subtrees as non-terminal nodes. Then we divide the whole tree structure into multiple "sequences" from top to bottom according to the non-terminal nodes. We then use sequence decoding to generate the tree structure in order. And for each sequence decoding process, we will feed the embedding of its parent node and sibling node as auxiliary input.

.. code:: python

    cur_index = 0
    while (cur_index <= max_index):
        if cur_index > max_dec_tree_depth:
            break
        ...
        # get parent and sibling embeddings.
        # do sequence decoding.
        ...

        cur_index = cur_index + 1

Where ``max_index`` is the number of non-terminal nodes and ``max_dec_tree_depth`` is the maximum number of non-terminal nodes allowed.

