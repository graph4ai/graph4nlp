import torch

from ...modules.graph_construction.base import DynamicGraphConstructionBase
from ...modules.utils.vocab_utils import VocabModel
from ...modules.utils.padding_utils import pad_2d_vals_no_size


if __name__ == '__main__':
    raw_text_data = [['I like nlp.', 'Same here!'], ['I like graph.', 'Same here!']]

    vocab_model = VocabModel(raw_text_data, max_word_vocab_size=None,
                                min_word_vocab_freq=1,
                                word_emb_size=300)

    embedding_styles = {'word_emb_type': 'w2v',
                        'node_edge_level_emb_type': 'bilstm',
                        'graph_level_emb_type': 'bilstm',
                        }
    dyn_graph_constructor = DynamicGraphConstructionBase(vocab_model.word_vocab, embedding_styles, 128)
    print('Passed DynamicGraphConstructionBase testing')
