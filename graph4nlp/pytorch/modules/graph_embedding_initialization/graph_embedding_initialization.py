from abc import abstractmethod
import torch.nn as nn

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import (
    EmbeddingConstruction,
)


class GraphEmbeddingInitialization(nn.Module):
    def __init__(
        self,
        word_vocab,
        embedding_style,
        hidden_size=None,
        fix_word_emb=True,
        fix_bert_emb=True,
        word_dropout=None,
        rnn_dropout=None,
    ):
        super(GraphEmbeddingInitialization, self).__init__()
        self.embedding_layer = EmbeddingConstruction(
            word_vocab,
            embedding_style["single_token_item"],
            emb_strategy=embedding_style["emb_strategy"],
            hidden_size=hidden_size,
            num_rnn_layers=embedding_style.get("num_rnn_layers", 1),
            fix_word_emb=fix_word_emb,
            fix_bert_emb=fix_bert_emb,
            bert_model_name=embedding_style.get("bert_model_name", "bert-base-uncased"),
            bert_lower_case=embedding_style.get("bert_lower_case", True),
            word_dropout=word_dropout,
            rnn_dropout=rnn_dropout,
        )

    @abstractmethod
    def forward(self, graph_data: GraphData):
        return self.embedding_layer(graph_data)
