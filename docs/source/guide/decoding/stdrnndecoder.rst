.. _std-rnn-decoder:

Chapter 5.1 Standard RNN Decoder
================================

There are various solutions for translating the graph based data to sequential outputs, such as RNN based and transformer based decoding.
However, in this section, we focus on the RNN based decoding mechanism.
Similar to most sequence-to-sequence decoder, the graph based ``StdRNNDecoder`` adopts the attention mechanism to learn
the soft alignment between the input graphs and the sequential outputs.

Specifically, we give a simple example on how ``StdRNNDecoder`` is initialized as follows,

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

Some advanced mechanisms
------------------------

Copy and coverage
`````````````````
To further enhance the performance, we also implement the ``copy`` and ``coverage`` mechanism.  

For ``copy`` mechanism, it helps model to copy words directly from the source sequence, and computed as, 
:math:`p(w) = p_{gen}  p_{softmax}(w) + (1 - p_{gen})  p_{copy}(w)`. We refer to the implement of `pointer-network <https://arxiv.org/abs/1506.03134>`_. Technically, for a certain mini-batch graphdata, we extend the original vocabulary to a full-vocabulary containing all words (including out-of-vocabulary (oov) words) in the mini-batch:

.. code:: python

    # First pick out all out-of-vocabulary (oov) words in the mini-batch graphdata.
    token_matrix = batch_graph.node_features["token_id"].squeeze(1)
    unk_index = (token_matrix == oov_dict.UNK).nonzero(as_tuple=False).squeeze(1).detach().cpu().numpy()
    unk_token = [batch_graph.node_attributes[index]["token"] for index in unk_index]

    # Second build the oov vocabulary.
    oov_dict = copy.deepcopy(vocab.in_word_vocab)
    oov_dict._add_words(unk_token)
    token_matrix_oov = token_matrix.clone()
    for idx in unk_index:
        unk_token = batch_graph.node_attributes[idx]["token"]
        oov_dict._add_words(unk_token)
        token_matrix_oov[idx] = oov_dict.getIndex(unk_token)
    batch_graph.node_features["token_id_oov"] = token_matrix_oov

Users can refer to the API ``prepare_ext_vocab``.
After that, the decoder learns the conditional probability of an output sequence with elements that are discrete tokens corresponding to positions in an input sequence. Code snippets as follows help with how it works.

.. code:: python

    if self.use_copy:
        pgen_collect = [dec_emb, hidden, attn_ptr]

        # the probability of copying a word from the source
        prob_ptr = torch.sigmoid(self.ptr(torch.cat(pgen_collect, -1)))

        # the probability of generating a word over the standard softmax on vocabulary model.
        prob_gen = 1 - prob_ptr 
        gen_output = torch.softmax(decoder_output, dim=-1)

        ret = prob_gen * gen_output
        need_pad_length = oov_dict.get_vocab_size() - self.vocab.get_vocab_size()
        output = torch.cat((ret, ret.new_zeros((batch_size, need_pad_length))), dim=1)

        # attention scores
        ptr_output = dec_attn_scores
        output.scatter_add_(1, src_seq, prob_ptr * ptr_output)
        decoder_output = output
    else:
        decoder_output = torch.softmax(decoder_output, dim=-1)

The returned ``decoder_output`` is a distribution over the extend dictionary ``oov_dict`` if ``copy`` is adopted. Users can set ``use_copy`` to ``True`` to use this feature. And the oov vocabulary must be passed when utilizing it.

As for the ``coverage`` mechanism, we refer to the paper: `Modeling Coverage for Neural Machine Translation <https://arxiv.org/abs/1601.04811>`_. Compared to the original attention that ignore the past alignment history information, we maintain a coverage vector to keep trace of that. As a result, it usually prevents the generated words and focuses on more about un-predicted words.
Users can easily use this feature by setting ``use_coverage`` to ``True``. Note that an additional coverage loss should be included when conducting backward propagating:

.. code:: python

    _, enc_attn_weights, coverage_vectors = Graph2Seq(graph, tgt)

    target_length = len(enc_attn_weights)
    loss = 0
    for i in range(target_length):
        if coverage_vectors[i] is not None:
            coverage_loss = torch.sum(
                torch.min(coverage_vectors[i], enc_attn_weights[i])) / coverage_vectors[-1].shape[0] * self.cover_loss
            loss += coverage_loss
    coverage_loss = loss / target_length

Users can use the ``CoverageLoss`` or ``Graph2SeqLoss`` (including both ``CoverageLoss`` and ``NLLLoss``) to conduct this process.



Separate attention
``````````````````
And different from most sequence-to-sequence decoder, our ``StdRNNDecoder`` designs both ``separate attention`` and ``uniform attention`` for sequential encoder's outputs :math:`\mathbf{S}` and graph encoder's outputs :math:`\mathcal{G}(\mathcal{V}, \mathcal{E})`, respectively:

1. Uniform attention: It means the decoder only attends on the graph encoder's output :math:`\mathcal{G}(\mathcal{V}, \mathcal{E})`. Note that we only support attending on node features and leaving the edges for future. Technically, it regard the nodes as tokens and apply attention on them to calculate the output vector. Users can set ``attention_type`` to ``uniform`` to use this feature.

2. Separate attention: There are two kinds of separate attention in this implement: a) attends on sequential encoder's outputs and graph encoder's outputs separately, b) attends on the graph's nodes separately.

* 2.1 Case a): The decoder first attends on :math:`\mathbf{S}` and :math:`\mathcal{G}(\mathcal{V}, \mathcal{E})` separately. Then it fuse the obtained attention results into one single vector to generate next token. Users can set ``attention_type`` to ``sep_diff_encoder_type`` to use this feature.

* 2.2 Case b): This feature is designed for heterogeneous graphs. Firstly, the decoder will group the nodes by their node types. Secondly, the decoder will attends on each group separately to obtain the vector representations. Finally, it will fuses the obtained vectors into one single vector. Users can set ``attention_type`` to ``sep_diff_node_type`` to use this feature. Specifically, ``node_type_num`` should be the amount of node types.

