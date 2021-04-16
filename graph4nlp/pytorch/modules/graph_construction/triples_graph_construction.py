import copy
import json

import torch
from stanfordcorenlp import StanfordCoreNLP

from .base import StaticGraphConstructionBase
from ...data.data import GraphData, to_batch


class TriplesBasedGraphConstruction(StaticGraphConstructionBase):
    """
        Triples based graph construction class

    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``single_token_item``, ``emb_strategy``, ``num_rnn_layers``, ``bert_model_name`` and ``bert_lower_case``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(self, embedding_style, vocab, hidden_size=300, fix_word_emb=True, fix_bert_emb=True, word_dropout=None,
                 rnn_dropout=None, device=None):
        super(TriplesBasedGraphConstruction, self).__init__(word_vocab=vocab,
                                                            embedding_styles=embedding_style,
                                                            hidden_size=hidden_size,
                                                            fix_word_emb=fix_word_emb,
                                                            fix_bert_emb=fix_bert_emb,
                                                            word_dropout=word_dropout,
                                                            rnn_dropout=rnn_dropout,
                                                            device=device)
        self.vocab = vocab
        self.verbase = 1
        self.device = self.embedding_layer.device

    def add_vocab(self, g):
        """
            Add node tokens appeared in graph g to vocabulary.

        Parameters
        ----------
        g: GraphData
            Graph data-structure.

        """
        for i in range(g.get_node_num()):
            attr = g.get_node_attrs(i)[i]
            self.vocab.word_vocab._add_words([attr["token"]])

    # def forward(self, batch_graphdata: list):
    #     node_size = []
    #     num_nodes = []
    #     num_word_nodes = []  # number of nodes that are extracted from the raw text in each graph
    #
    #     for g in batch_graphdata:
    #         g.to(self.device)
    #         g.node_features['token_id'] = g.node_features['token_id'].to(self.device)
    #         num_nodes.append(g.get_node_num())
    #         # num_word_nodes.append(len([1 for i in range(len(g.node_attributes)) if g.node_attributes[i]['type'] == 0]))
    #         num_word_nodes.append(len([1 for i in range(len(g.node_attributes))]))
    #         node_size.extend([1 for i in range(num_nodes[-1])])
    #
    #     batch_gd = to_batch(batch_graphdata)
    #     b_node = batch_gd.get_node_num()
    #     assert b_node == sum(num_nodes), print(b_node, sum(num_nodes))
    #     node_size = torch.Tensor(node_size).to(self.device).int()
    #     num_nodes = torch.Tensor(num_nodes).to(self.device).int()
    #     num_word_nodes = torch.Tensor(num_word_nodes).to(self.device).int()
    #     node_emb = self.embedding_layer(batch_gd, node_size, num_nodes, num_word_items=num_word_nodes)
    #     batch_gd.node_features["node_feat"] = node_emb
    #
    #     return batch_gd

    def forward(self, batch_graphdata: list):
        batch_graphdata = self.embedding_layer(batch_graphdata)
        return batch_graphdata

    def embedding(self, node_attributes, edge_attributes):
        node_emb, edge_emb = self.embedding_layer(
            node_attributes, edge_attributes)
        return node_emb, edge_emb
