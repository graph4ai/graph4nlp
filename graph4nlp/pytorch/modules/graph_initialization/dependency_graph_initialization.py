from .base import GraphInitializationBase


class DependencyBasedGraphInitialization(GraphInitializationBase):
    def __init__(self, word_vocab,
                 embedding_style,
                 hidden_size=None,
                 fix_word_emb=True,
                 fix_bert_emb=True,
                 word_dropout=None,
                 rnn_dropout=None,):
        super(DependencyBasedGraphInitialization, self).__init__(word_vocab=word_vocab,
                                                                 embedding_styles=embedding_style,
                                                                 hidden_size=hidden_size,
                                                                 fix_word_emb=fix_word_emb,
                                                                 fix_bert_emb=fix_bert_emb,
                                                                 word_dropout=word_dropout,
                                                                 rnn_dropout=rnn_dropout)

    def forward(self, graph_data):
        return self.embedding_layer(graph_data)
