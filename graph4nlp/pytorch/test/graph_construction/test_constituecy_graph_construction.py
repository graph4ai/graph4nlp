import random
import numpy as np
import torch
from stanfordcorenlp import StanfordCoreNLP

from ...modules.graph_construction.constituency_graph_construction import (
    ConstituencyBasedGraphConstruction,
)
from ...modules.utils.vocab_utils import VocabModel

if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # some tests for data structure code.

    # g = GraphData()
    # g.add_nodes(1)
    # g.add_nodes(1)
    # print(g.node_attributes[1]['node_attr'])
    # print(g.get_node_attrs(1))
    # g.node_attributes[1]['word'] = "word"
    # print(g.get_node_attrs(1))

    # print(g.node_features)
    # g.node_features['node_feat'] = torch.zeros(g.get_node_num(), 128)
    # print(g.node_features)
    # g.set_node_features(1, {"node_feat":torch.ones(1, 128)})
    # print(g.get_node_features(1))

    # g.add_edge(0,1)
    # g.add_edge(0,1)
    # print(g.get_edge_num())
    # print(g.edges)
    # print(list(g.nodes))
    # raise BaseException

    raw_data = "I love you. My motherland."
    # raw_data = "James went to the corner-shop. He want to buy some (eggs),
    #  <milk> and bread for breakfast."
    vocab_model = VocabModel(
        raw_data, max_word_vocab_size=None, min_word_vocab_freq=1, word_emb_size=300
    )

    embedding_styles = {
        "word_emb_type": "w2v",
        "node_edge_level_emb_type": "mean",
        "graph_level_emb_type": "identity",
    }

    nlp_parser = StanfordCoreNLP("http://localhost", port=9000, timeout=300000)
    print("syntactic parser ready\n-------------------")

    # constituency_graph_gonstructor = ConstituencyBasedGraphConstruction(
    # hidden_emb_size=128, embedding_style=embedding_styles,
    # word_emb_size=300, vocab=vocab_model.word_vocab)
    # for sentence in raw_data:
    # output_graph = constituency_graph_gonstructor.forward(sentence[0], nlp_parser)
    output_graph = ConstituencyBasedGraphConstruction.topology(raw_data, nlp_parser)
    print(output_graph.node_attributes)
    print(output_graph.edges)
    for _edge in output_graph.get_all_edges():
        print(
            output_graph.nodes[_edge[0]].attributes["token"],
            output_graph.nodes[_edge[1]].attributes["token"],
        )
    # print("-----------------------\nvocab size")
    # print(vocab_model.word_vocab.word2index)
