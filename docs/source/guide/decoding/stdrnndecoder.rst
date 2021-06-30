.. _std-rnn-decoder:

Chapter 5.1 Standard RNN Decoder
=====================

There are various solutions for translating the graph based data to sequential outputs, such as RNN based and transformer based decoding.
However, in this section, we focus on the RNN based decoding mechanism.
Similar to most sequence-to-sequence decoder, the graph based ``StdRNNDecoder`` adopts the attention mechanism to learn
the soft alignment between the input graphs and the sequential outputs.
Furthermore, to further enhance the performance, we also implement the ``copy`` and ``coverage`` mechanism.

Different from most sequence-to-sequence decoder, our ``StdRNNDecoder`` designs both separate attention and uniform attention
for sequential encoder's outputs :math:`\mathbf{S}` and graph encoder's outputs :math:`\mathcal{G}(\mathcal{V}, \mathcal{E})`, respectively:

1. Uniform attention: It means the decoder only attends on the graph encoder's output :math:`\mathcal{G}(\mathcal{V}, \mathcal{E})`. Note that we only support attending on node features and leaving the edges for future. Technically, it regard the nodes as tokens and apply attention on them to calculate the output vector. Users can set ``attention_type`` to ``uniform`` to use this feature.

2. Separate attention: There are two kinds of separate attention in this implement: a) attends on sequential encoder's outputs and graph encoder's outputs separately, b) attends on the graph's nodes separately.

* 2.1 Case a): The decoder first attends on :math:`\mathbf{S}` and :math:`\mathcal{G}(\mathcal{V}, \mathcal{E})` separately. Then it fuse the obtained attention results into one single vector to generate next token. Users can set ``attention_type`` to ``sep_diff_encoder_type`` to use this feature.

* 2.2 Case b): This feature is designed for heterogeneous graphs. Firstly, the decoder will group the nodes by their node types. Secondly, the decoder will attends on each group separately to obtain the vector representations. Finally, it will fuses the obtained vectors into one single vector. Users can set ``attention_type`` to ``sep_diff_node_type`` to use this feature. Specifically, ``node_type_num`` should be the amount of node types.

.. code:: python

    from graph4nlp.pytorch.datasets.jobs import JobsDataset
    from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
    from graph4nlp.pytorch.modules.config import get_basic_args
    from graph4nlp.pytorch.models.graph2seq import Graph2Seq
    from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config

    # define your vocab_model

    dec_word_emb = WordEmbedding(vocab_model.out_word_vocab.embeddings.shape[0],
                                 vocab_model.out_word_vocab.embeddings.shape[1],
                                 pretrained_word_emb=vocab_model.out_word_vocab.embeddings,
                                 fix_emb=fix_word_emb).word_emb_layer

    attention_type = "uniform"

    seq_decoder = StdRNNDecoder(rnn_type="lstm", max_decoder_step=50,
                                attention_type=attention_type, fuse_strategy="average",
                                use_coverage=False, use_copy=False,
                                input_size=1024, hidden_size=1024, rnn_emb_input_size=1024,
                                graph_pooling_strategy="max",
                                word_emb=dec_word_emb, vocab=vocab_model.out_word_vocab)

    # g is the output of encoder, tgt is the ground truth sequence
    predicted = seq_decoder(batch_graph=batch_graph, tgt_seq=tgt_seq)