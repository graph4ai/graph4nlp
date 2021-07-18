.. _guide-embedding_construction:

Embedding Construction
===========


The embedding construction module aims to learn the initial node/edge embeddings for the input graph
before being consumed by the subsequent GNN model.



EmbeddingConstruction
--------------------------------------


The ``EmbeddingConstruction`` class inherits the ``EmbeddingConstructionBase`` base class, and supports
various strategies (e.g., word2vec, BiLSTM, BERT) for initializing both single-token (i.e., containing single token)
or multi-token (i.e., containing multiple tokens) nodes/edges.

As shown in the below code piece,
users can specify whether the node/edge contains single token or multiple tokens by setting ``single_token_item``.
``emb_strategy`` is probably the most important parameter in this class which specifies which the encoding
strategy. For instance, for single-token node/edge, ``w2v_bilstm`` strategy means we first use word2vec embeddings
to initialize each item, and then apply a BiLSTM encoder to encode the whole graph (assuming the node order reserves
the sequential order in raw text).
For multi-token node/edge, ``w2v_bilstm`` strategy means we first use the word2vec embeddings to initialize
each token in the node/edge, then apply a BiLSTM encoder to encode each node/edge text.



.. code-block:: python

    class EmbeddingConstruction(EmbeddingConstructionBase):
        def __init__(self,
                        word_vocab,
                        single_token_item,
                        emb_strategy='w2v_bilstm',
                        hidden_size=None,
                        num_rnn_layers=1,
                        fix_word_emb=True,
                        fix_bert_emb=True,
                        bert_model_name='bert-base-uncased',
                        bert_lower_case=True,
                        word_dropout=None,
                        bert_dropout=None,
                        rnn_dropout=None):
            super(EmbeddingConstruction, self).__init__()
            self.word_dropout = word_dropout
            self.bert_dropout = bert_dropout
            self.rnn_dropout = rnn_dropout
            self.single_token_item = single_token_item

            assert emb_strategy in ('w2v', 'w2v_bilstm', 'w2v_bigru',
                            'bert', 'bert_bilstm', 'bert_bigru',
                            'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru'),\
                "emb_strategy must be one of ('w2v', 'w2v_bilstm', 'w2v_bigru', 'bert', 'bert_bilstm', 'bert_bigru', 'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru')"

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


            word_emb_size = 0
            self.word_emb_layers = nn.ModuleDict()
            if 'w2v' in word_emb_type:
                self.word_emb_layers['w2v'] = WordEmbedding(
                                word_vocab.embeddings.shape[0],
                                word_vocab.embeddings.shape[1],
                                pretrained_word_emb=word_vocab.embeddings,
                                fix_emb=fix_word_emb)
                word_emb_size += word_vocab.embeddings.shape[1]

            if 'node_edge_bert' in word_emb_type:
                self.word_emb_layers['node_edge_bert'] = BertEmbedding(name=bert_model_name,
                                                                fix_emb=fix_bert_emb,
                                                                lower_case=bert_lower_case)
                word_emb_size += self.word_emb_layers['node_edge_bert'].bert_model.config.hidden_size

            if 'seq_bert' in word_emb_type:
                self.word_emb_layers['seq_bert'] = BertEmbedding(name=bert_model_name,
                                                                fix_emb=fix_bert_emb,
                                                                lower_case=bert_lower_case)

            if node_edge_emb_strategy in ('bilstm', 'bigru'):
                self.node_edge_emb_layer = RNNEmbedding(
                                        word_emb_size,
                                        hidden_size,
                                        bidirectional=True,
                                        num_layers=num_rnn_layers,
                                        rnn_type='lstm' if node_edge_emb_strategy == 'bilstm' else 'gru',
                                        dropout=rnn_dropout)
                rnn_input_size = hidden_size
            elif node_edge_emb_strategy == 'mean':
                self.node_edge_emb_layer = MeanEmbedding()
                rnn_input_size = word_emb_size
            else:
                rnn_input_size = word_emb_size

            if 'seq_bert' in word_emb_type:
                rnn_input_size += self.word_emb_layers['seq_bert'].bert_model.config.hidden_size

            if seq_info_encode_strategy in ('bilstm', 'bigru'):
                self.output_size = hidden_size
                self.seq_info_encode_layer = RNNEmbedding(
                                        rnn_input_size,
                                        hidden_size,
                                        bidirectional=True,
                                        num_layers=num_rnn_layers,
                                        rnn_type='lstm' if seq_info_encode_strategy == 'bilstm' else 'gru',
                                        dropout=rnn_dropout)

            else:
                self.output_size = rnn_input_size
                self.seq_info_encode_layer = None





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
        def __init__(self,
                    name='bert-base-uncased',
                    max_seq_len=500,
                    doc_stride=250,
                    fix_emb=True,
                    lower_case=True):
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

