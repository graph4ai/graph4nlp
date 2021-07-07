.. _guide-embedding_construction:

Embedding Construction
===========


The embedding construction module aims to learn the initial node/edge embeddings for the input graph
before being consumed by the subsequent GNN model.
Below is an example to call the embedding construction API.

.. code-block:: python

    from graph4nlp.pytorch.modules.graph_construction.embedding_construction import EmbeddingConstruction

    embedding_layer = EmbeddingConstruction(word_vocab, # word vocab instance
                                True, # whether a single_token item or not
                                emb_strategy='w2v_bert_bilstm',
                                hidden_size=256,
                                num_rnn_layers=1,
                                fix_word_emb=True,
                                fix_bert_emb=True,
                                bert_model_name='bert-base-uncased',
                                bert_lower_case=True,
                                word_dropout=0.4,
                                rnn_dropout=0.4)

The `emb_strategy` argument specifies whether we apply pretrained word vectors, RNN (e.g., BiLSTM) or BERT when encoding the node/edge features.
Various options for `emb_strategy` are supported, including 'w2v', 'w2v_bilstm', 'w2v_bert_bilstm' and so on.


