.. _guide-embedding_construction:

Embedding Construction
===========


The embedding construction module aims to learn the initial node/edge embeddings for the input graph
before being consumed by the subsequent GNN model.



EmbeddingConstruction
--------------------------------------

The ``EmbeddingConstruction`` class supports various strategies for initializing both single-token
(i.e., containing single token) and multi-token (i.e., containing multiple tokens) items (i.e., node/edge).
As shown in the below code piece, for both single-token and multi-token items, supported embedding strategies
include `w2v`, `w2v_bilstm`, `w2v_bigru`, `bert`, `bert_bilstm`, `bert_bigru`, `w2v_bert`, `w2v_bert_bilstm`
and `w2v_bert_bigru`.


.. code-block:: python

    assert emb_strategy in ('w2v', 'w2v_bilstm', 'w2v_bigru', 'bert', 'bert_bilstm', 'bert_bigru',
        'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru')

    word_emb_type = set()
    if single_token_item:
        node_edge_emb_strategy = None
        if 'w2v' in emb_strategy:
            word_emb_type.add('w2v')

        if 'bert' in emb_strategy:
            word_emb_type.add('seq_bert')

        if 'bilstm' in emb_strategy:
            seq_info_encode_strategy = 'bilstm'
        elif 'bigru' in emb_strategy:
            seq_info_encode_strategy = 'bigru'
        else:
            seq_info_encode_strategy = 'none'
    else:
        seq_info_encode_strategy = 'none'
        if 'w2v' in emb_strategy:
            word_emb_type.add('w2v')

        if 'bert' in emb_strategy:
            word_emb_type.add('node_edge_bert')

        if 'bilstm' in emb_strategy:
            node_edge_emb_strategy = 'bilstm'
        elif 'bigru' in emb_strategy:
            node_edge_emb_strategy = 'bigru'
        else:
            node_edge_emb_strategy = 'mean'




For instance, for single-token item, ``w2v_bilstm`` strategy means we first use word2vec embeddings
to initialize each item, and then apply a BiLSTM encoder to encode the whole graph (assuming the node
order reserves the sequential order in raw text). Compared to ``w2v_bilstm``, the ``w2v_bert_bilstm``
strategy in addition applies the BERT encoder to the whole graph (i.e., sequential text), the concatenation of
the BERT embedding and word2vec embedding instead of word2vec embedding will be fed into the BiLSTM encoder.


.. code-block:: python

    # single-token item graph
    feat = []
    token_ids = batch_gd.batch_node_features["token_id"]
    if 'w2v' in self.word_emb_layers:
        word_feat = self.word_emb_layers['w2v'](token_ids).squeeze(-2)
        word_feat = dropout_fn(word_feat, self.word_dropout, shared_axes=[-2], training=self.training)
        feat.append(word_feat)

    new_feat = feat
    if 'seq_bert' in self.word_emb_layers:
        gd_list = from_batch(batch_gd)
        raw_tokens = [[gd.node_attributes[i]['token'] for i in range(gd.get_node_num())] for gd in gd_list]
        bert_feat = self.word_emb_layers['seq_bert'](raw_tokens)
        bert_feat = dropout_fn(bert_feat, self.bert_dropout, shared_axes=[-2], training=self.training)
        new_feat.append(bert_feat)

    new_feat = torch.cat(new_feat, -1)
    if self.seq_info_encode_layer is None:
        batch_gd.batch_node_features["node_feat"] = new_feat
    else:
        rnn_state = self.seq_info_encode_layer(new_feat, torch.LongTensor(batch_gd._batch_num_nodes).to(batch_gd.device))
        if isinstance(rnn_state, (tuple, list)):
            rnn_state = rnn_state[0]

        batch_gd.batch_node_features["node_feat"] = rnn_state




For multi-token item, ``w2v_bilstm`` strategy means we first use the word2vec embeddings to initialize
each token in the item, then apply a BiLSTM encoder to encode each item text. Compared to ``w2v_bilstm``,
the ``w2v_bert_bilstm`` strategy in addition applies the BERT encoder to each item text, the concatenation of
the BERT embedding and word2vec embedding instead of word2vec embedding will be fed into the BiLSTM encoder.


.. code-block:: python

    # multi-token item graph
    feat = []
    token_ids = batch_gd.node_features["token_id"]
    if 'w2v' in self.word_emb_layers:
        word_feat = self.word_emb_layers['w2v'](token_ids)
        word_feat = dropout_fn(word_feat, self.word_dropout, shared_axes=[-2], training=self.training)
        feat.append(word_feat)

    if 'node_edge_bert' in self.word_emb_layers:
        input_data = [batch_gd.node_attributes[i]['token'].strip().split(' ') for i in range(batch_gd.get_node_num())]
        node_edge_bert_feat = self.word_emb_layers['node_edge_bert'](input_data)
        node_edge_bert_feat = dropout_fn(node_edge_bert_feat, self.bert_dropout, shared_axes=[-2], training=self.training)
        feat.append(node_edge_bert_feat)

    if len(feat) > 0:
        feat = torch.cat(feat, dim=-1)
        node_token_lens = torch.clamp((token_ids != Vocab.PAD).sum(-1), min=1)
        feat = self.node_edge_emb_layer(feat, node_token_lens)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]

        feat = batch_gd.split_features(feat)

    batch_gd.batch_node_features["node_feat"] = feat




Various embedding modules
--------------------------------------

Various embedding modules are provided in the library to support embedding construction.
For instance, ``WordEmbedding`` class aims to convert the input word index sequence to the word embedding matrix.
``MeanEmbedding`` class simply computes the average embeddings.
``RNNEmbedding`` class applies the RNN network (e.g., GRU, LSTM, BiGRU, BiLSTM) to a sequence of word embeddings.

We will introduce ``BertEmbedding`` in more detail next.
``BertEmbedding`` class calls the Hugging Face Transformers APIs to compute the BERT embeddings for the input text.
Transformer-based models like BERT have limit on the maximal sequence length.
The ``BertEmbedding`` class can automaticall cut the long input sequence to multiple small chunks and
call Transformers APIs for each of the small chunk, and then automtically merge their embeddings to
obtain the embedding for the original long sequence.
Below is the code piece showing the ``BertEmbedding`` class API. Users can specify ``max_seq_len`` and ``doc_stride``
to indicate the maximal sequence length and the stride (i.e., similar to the stride idea in ConvNet)
when cutting long text into small chunks.
In addition, instead of returning the last encoder layer as the output state, it returns the weighted average of
all the encoder layer states as the output layer, as we find this works better in practice. Note the weight is a
learnable parameter.


.. code-block:: python

    class BertEmbedding(nn.Module):
        def __init__(self, name='bert-base-uncased', max_seq_len=500, doc_stride=250, fix_emb=True, lower_case=True):
            super(BertEmbedding, self).__init__()
            self.bert_max_seq_len = max_seq_len
            self.bert_doc_stride = doc_stride
            self.fix_emb = fix_emb

            from transformers import BertModel
            from transformers import BertTokenizer
            print('[ Using pretrained BERT embeddings ]')
            self.bert_tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=lower_case)
            self.bert_model = BertModel.from_pretrained(name)
            if fix_emb:
                print('[ Fix BERT layers ]')
                self.bert_model.eval()
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            else:
                print('[ Finetune BERT layers ]')
                self.bert_model.train()

            # compute weighted average over BERT layers
            self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, self.bert_model.config.num_hidden_layers)))

