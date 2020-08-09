import time
import numpy as np
import torch
import dgl

from ...data.data import GraphData
from ...modules.graph_construction.embedding_construction import EmbeddingConstruction
from ...modules.utils.vocab_utils import VocabModel
from ...modules.utils.padding_utils import pad_2d_vals, pad_2d_vals_no_size


if __name__ == '__main__':
    raw_text_data = [['Google News is a news aggregator service developed by Google.',
                    'It presents a continuous flow of articles organized from thousands of publishers and magazines.'],
                    ['Google News is available as an app on Android, iOS, and the Web.',
                     'Google released a beta version in September 2002 and the official app in January 2006.']]
    vocab_model = VocabModel(raw_text_data, max_word_vocab_size=None,
                            min_word_vocab_freq=1,
                            pretrained_word_emb_file=None,
                            word_emb_size=300)

    batch_size = 64
    hidden_size = 128
    num_nodes = list(range(3, 3 + batch_size))
    np.random.shuffle(num_nodes)

    # build graph
    max_node_len = 0
    graph_list = []
    node_size = []
    for i in range(batch_size):
        graph = GraphData()
        graph.add_nodes(num_nodes[i])
        for j in range(num_nodes[i]):
            tokens = np.random.choice(list(vocab_model.word_vocab.word2index.keys()), np.random.choice(range(1, 7)))
            graph.node_attributes[j]['token_idx'] = vocab_model.word_vocab.to_index_sequence_for_list(tokens)
            node_size.append(len(graph.node_attributes[j]['token_idx']))
            max_node_len = max(node_size[-1], max_node_len)

        graph_list.append(graph)

    for graph in graph_list:
        tmp_token_idx = [graph.node_attributes[j]['token_idx'] for j in range(graph.get_node_num())]
        tmp_token_idx = torch.LongTensor((pad_2d_vals(tmp_token_idx, len(tmp_token_idx), max_node_len)))
        graph.node_features['token_idx'] = tmp_token_idx

    graph_list = [graph.to_dgl() for graph in graph_list]
    bg = dgl.batch(graph_list, edge_attrs=None)
    node_size = torch.LongTensor(node_size)
    num_nodes = torch.LongTensor(num_nodes)

    emb_constructor = EmbeddingConstruction(vocab_model.word_vocab, 'w2v', 'bilstm', 'bilstm', hidden_size, device=None)
    t0 = time.time()
    node_feat = emb_constructor(bg.ndata['token_idx'], node_size, num_nodes)
    print('runtime: {}'.format(time.time() - t0))
    print('mean', node_feat.mean())

    print('emb: {}'.format(node_feat.shape))
