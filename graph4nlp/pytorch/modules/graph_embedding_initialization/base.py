import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding_construction import EmbeddingConstruction


class GraphEmbeddingInitializationBase(nn.Module):
    def __init__(self, word_vocab,
                 embedding_styles,
                 hidden_size=None,
                 fix_word_emb=True,
                 fix_bert_emb=True,
                 word_dropout=None,
                 rnn_dropout=None,):
        super(GraphEmbeddingInitializationBase, self).__init__()
        self.embedding_layer = EmbeddingConstruction(
            word_vocab,
            embedding_styles["single_token_item"],
            emb_strategy=embedding_styles["emb_strategy"],
            hidden_size=hidden_size,
            num_rnn_layers=embedding_styles.get("num_rnn_layers", 1),
            fix_word_emb=fix_word_emb,
            fix_bert_emb=fix_bert_emb,
            bert_model_name=embedding_styles.get("bert_model_name", "bert-base-uncased"),
            bert_lower_case=embedding_styles.get("bert_lower_case", True),
            word_dropout=word_dropout,
            rnn_dropout=rnn_dropout,
        )