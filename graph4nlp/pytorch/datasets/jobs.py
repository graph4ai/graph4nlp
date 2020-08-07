import os

import stanfordcorenlp
import torch
import numpy as np

# from ..data.dataset import DependencyDataset
from graph4nlp.pytorch.data.dataset import DependencyDataset
# from ..modules.graph_construction.dependency_graph_construction import \
#     DependencyBasedGraphConstruction as topology_builder
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel, Vocab

dataset_root = '../test/dataset/jobs'


class SymbolsManager():
    def __init__(self, whether_add_special_tags):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0
        self.whether_add_special_tags = whether_add_special_tags
        if whether_add_special_tags:
            # PAD: padding token = 0
            self.add_symbol('<P>')
            # GO: start token = 1
            self.add_symbol('<S>')
            # EOS: end token = 2
            self.add_symbol('<E>')
            # UNK: unknown token = 3
            self.add_symbol('<U>')
            # NON: non-terminal token = 4
            self.add_symbol('<N>')

    def add_symbol(self, s):
        if s not in self.symbol2idx:
            self.symbol2idx[s] = self.vocab_size
            self.idx2symbol[self.vocab_size] = s
            self.vocab_size += 1
        return self.symbol2idx[s]

    def get_symbol_idx(self, s):
        if s not in self.symbol2idx:
            if self.whether_add_special_tags:
                return self.symbol2idx['<U>']
            else:
                print("not reached!")
                return 0
        return self.symbol2idx[s]

    def get_idx_symbol(self, idx):
        if idx not in self.idx2symbol:
            return '<U>'
        return self.idx2symbol[idx]

    def init_from_file(self, fn, min_freq, max_vocab_size):
        # the vocab file is sorted by word_freq
        print("loading vocabulary file: {}".format(fn))
        with open(fn, "r") as f:
            for line in f:
                l_list = line.strip().split('\t')
                c = int(l_list[1])
                if c >= min_freq:
                    self.add_symbol(l_list[0])
                if self.vocab_size > max_vocab_size:
                    break

    def get_symbol_idx_for_list(self, l):
        r = []
        for i in range(len(l)):
            r.append(self.get_symbol_idx(l[i]))
        return r


class JobsDataset(DependencyDataset):
    @property
    def raw_file_names(self) -> list:
        return ['test.txt', 'train.txt', 'vocab.f.txt', 'vocab.q.txt']

    @property
    def processed_file_names(self) -> list:
        return ['graph.pt']

    @property
    def topology_file_names(self):
        return ['topology.pt']

    @property
    def vocab_file_names(self):
        return ['vocabulary.pt']

    @property
    def preprocessed_file_names(self) -> list:
        return ['sequence.pt']

    def preprocess(self):
        form_manager = SymbolsManager(True)
        form_manager.init_from_file(os.path.join(self.raw_dir, 'vocab.f.txt'), 0, 15000)
        data = []
        with open("{}/{}.txt".format(self.raw_dir, "train"), "r") as f:
            for line in f:
                l_list = line.split("\t")
                w_list = l_list[0]
                r_list = l_list[1]

                # r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
                data.append((w_list, r_list))
        self.seq_data = data
        torch.save(self.seq_data, os.path.join(self.processed_dir, 'sequence.pt'))

    def build_topology(self):
        processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)
        topo_data = []
        for item in self.seq_data:
            topo = self.topology_builder.topology(raw_text_data=item[0], nlp_processor=processor,
                                                  merge_strategy='tailhead', edge_strategy=None)
            topo_data.append(topo)
        self.topo_data = topo_data
        torch.save(self.topo_data, os.path.join(self.topology_subdir, 'topology.pt'))

    def build_vocab(self):
        from collections import Counter
        from graph4nlp.pytorch.modules.utils.vocab_utils import collect_vocabs, word_tokenize
        from graph4nlp.pytorch.data.data import GraphData

        token_counter = Counter()
        for graph in self.topo_data:
            graph: GraphData
            for i in range(graph.get_node_num()):
                node_attr = graph.get_node_attrs(i)[i]
                token = node_attr.get('token')
                token_counter.update([token])
        inst_list = []
        for item in self.seq_data:
            inst_list.append(item[1])

        q_counter = collect_vocabs(inst_list, word_tokenize)
        token_counter.update(q_counter)

        vocab_model = VocabModel(words_counter=token_counter, min_word_vocab_freq=1,
                                 word_emb_size=300)
        self.vocab_model = vocab_model
        pass

    def vectorization(self):
        padding_length = 20
        data = []
        for i in range(len(self.seq_data)):
            topo = self.topo_data[i]

            tgt = self.seq_data[i][1]
            tgt_token_id = self.vocab_model.word_vocab.to_index_sequence(tgt)
            tgt_token_id = np.array(tgt_token_id)
            tgt_token_id = torch.from_numpy(tgt_token_id)
            if tgt_token_id.shape[0] < padding_length:
                need_pad_length = padding_length - tgt_token_id.shape[0]
                pad = torch.zeros(need_pad_length).fill_(self.vocab_model.word_vocab.PAD)
                tgt_token_id = torch.cat((tgt_token_id, pad.long()), dim=0)
            data.append((topo, tgt_token_id))
        torch.save(data, os.path.join(self.processed_dir, 'graph.pkl'))
        pass

    def download(self):
        raise NotImplementedError(
            'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir='../test/dataset/jobs'):
        super(JobsDataset, self).__init__(root_dir)
        self.data = torch.load(os.path.join(self.processed_dir, 'graph.pkl'))

    def get(self, index: int):
        return self.data[index]

    def len(self):
        return len(self.data)

    def collate_fn(self, data_list):
        return None

if __name__ == '__main__':
    JobsDataset()