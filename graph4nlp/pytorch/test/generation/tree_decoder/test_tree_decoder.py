import random
import numpy as np
import torch
from stanfordcorenlp import StanfordCoreNLP

from ....modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder
from ....modules.utils.tree_utils import Tree, Vocab, DataLoader, to_cuda
from ....data.data import GraphData


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_hyper_para_dict = {}
    train_hyper_para_dict['src_vocab_file'] = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\vocab.q.txt"
    train_hyper_para_dict['tgt_vocab_file'] = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\vocab.f.txt"
    train_hyper_para_dict['data_file'] = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\train.txt"
    train_hyper_para_dict['mode'] = "train"
    train_hyper_para_dict['min_freq'] = 2
    train_hyper_para_dict['max_vocab_size'] = 10000
    train_hyper_para_dict['batch_size'] = 20
    train_hyper_para_dict['device'] = None

    train_data_loader = DataLoader(**train_hyper_para_dict)
    print("samples number: ", len(train_data_loader.data))

    # test_hyper_para_dict = {}
    # test_hyper_para_dict['src_vocab_file'] = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\vocab.q.txt"
    # test_hyper_para_dict['tgt_vocab_file'] = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\vocab.f.txt"
    # test_hyper_para_dict['data_file'] = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\test.txt"
    # test_hyper_para_dict['mode'] = "test"
    # test_hyper_para_dict['min_freq'] = 2
    # test_hyper_para_dict['max_vocab_size'] = 10000
    # test_hyper_para_dict['batch_size'] = 1
    # test_hyper_para_dict['device'] = None

    # test_data_loader = DataLoader(**test_hyper_para_dict)
    # print(len(test_data_loader.data))

    # embedding_styles = {
    #     'word_emb_type': 'w2v',
    #     'node_edge_level_emb_type': 'mean',
    #     'graph_level_emb_type': 'identity',
    # }

    # nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)
    # print("syntactic parser ready\n-------------------")

    # # constituency_graph_gonstructor = ConstituencyBasedGraphConstruction(hidden_emb_size=128, embedding_style=embedding_styles, word_emb_size=300, vocab=vocab_model.word_vocab)
    # # for sentence in raw_data:
    #     # output_graph = constituency_graph_gonstructor.forward(sentence[0], nlp_parser)
    # output_graph = ConstituencyBasedGraphConstruction.topology(raw_data, nlp_parser)
    # print(output_graph.node_attributes)
    # print(output_graph.edges)
    # print("-----------------------\nvocab size")
    # print(vocab_model.word_vocab.word2index)