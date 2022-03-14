import time
import numpy as np
import torch

from ...data.data import GraphData, to_batch
from ...data.dataset import DataItem
from ...modules.utils.padding_utils import pad_2d_vals
from ...modules.utils.vocab_utils import VocabModel
from ...modules.graph_embedding_initialization.graph_embedding_initialization import (
    GraphEmbeddingInitialization
)


class RawTextDataItem(DataItem):
    def __init__(self, input_text, tokenizer=None):
        """
        input_text: str
        """
        super(RawTextDataItem, self).__init__(input_text, tokenizer)

    def extract(self, lower_case=True):
        if lower_case:
            return self.input_text.lower().split()
        else:
            return self.input_text.split()


if __name__ == "__main__":
    raw_text_data = [
        "This is a news aggregator service.",
        "It presents a continuous flow of articles.",
        "This is available as an app on Android, iOS, and the Web.",
        "They released a beta version in September 2002.",
    ]
    raw_text_dataItem = [RawTextDataItem(text) for text in raw_text_data]
    vocab_model = VocabModel(
        raw_text_dataItem,
        max_word_vocab_size=None,
        min_word_vocab_freq=1,
        pretrained_word_emb_name=None,
        word_emb_size=300,
    )

    batch_size = 1
    hidden_size = 128
    num_nodes = list(range(3, 3 + batch_size))
    np.random.shuffle(num_nodes)

    embedding_style = {
        "single_token_item": False,
        "emb_strategy": "w2v_bilstm",
        "num_rnn_layers": 1,
        "bert_model_name": None,
        "bert_lower_case": None,
    }

    # build graph
    max_node_len = 0
    graph_list = []
    node_size = []
    for i in range(batch_size):
        graph = GraphData()
        graph.add_nodes(num_nodes[i])
        for j in range(num_nodes[i]):
            tokens = np.random.choice(
                list(vocab_model.in_word_vocab.word2index.keys()), np.random.choice(range(1, 7))
            )
            graph.node_attributes[j][
                "token_id"
            ] = vocab_model.in_word_vocab.to_index_sequence_for_list(tokens)
            node_size.append(len(graph.node_attributes[j]["token_id"]))
            max_node_len = max(node_size[-1], max_node_len)

        graph_list.append(graph)

    for graph in graph_list:
        tmp_token_idx = [graph.node_attributes[j]["token_id"] for j in range(graph.get_node_num())]
        tmp_token_idx = torch.LongTensor(
            (pad_2d_vals(tmp_token_idx, len(tmp_token_idx), max_node_len))
        )

        graph.node_features["token_id"] = tmp_token_idx

    bg = to_batch(graph_list)
    node_size = torch.LongTensor(node_size)
    num_nodes = torch.LongTensor(num_nodes)

    emb_constructor = GraphEmbeddingInitialization(
        word_vocab=vocab_model.in_word_vocab,
        embedding_style=embedding_style,
        hidden_size=hidden_size
    )

    t0 = time.time()
    node_feat = emb_constructor(bg).batch_node_features["node_feat"]
    print("runtime: {}".format(time.time() - t0))
    print("mean", node_feat.mean())
    print("emb: {}".format(node_feat.shape))
