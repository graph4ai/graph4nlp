import os

import numpy as np
import stanfordcorenlp
import torch

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.data.dataset import DependencyDataset
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel

dataset_root = '../test/dataset/jobs'


class JobsDataset(DependencyDataset):
    @property
    def raw_file_names(self) -> list:
        return ['sequence.pt']

    @property
    def processed_file_names(self) -> list:
        return ['graph.pt']

    @property
    def topology_file_names(self):
        return ['topology.pt']

    @property
    def vocab_file_names(self):
        return ['vocabulary.pt']

    def build_topology(self):
        self.seq_data = torch.load(os.path.join(self.raw_dir, 'sequence.pt'))
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
        padding_length = 50
        data = []

        for i in range(len(self.seq_data)):
            topo = self.topo_data[i]
            topo: GraphData
            token_matrix = []
            for node_idx in range(topo.get_node_num()):
                node_token = topo.get_node_attrs(node_idx)[node_idx]['token']
                node_token_id = self.vocab_model.word_vocab.getIndex(node_token)
                topo.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.Tensor(token_matrix).long()
            # topo.node_features['token_id'][:] = token_matrix
            topo.set_node_features(slice(None, None, None), {'token_id': token_matrix})

            tgt = self.seq_data[i][1]
            tgt_token_id = self.vocab_model.word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(self.vocab_model.word_vocab.EOS)
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

    @staticmethod
    def collate_fn(data_list):
        graph_data = [item[0] for item in data_list]
        tgt_seq = [item[1].unsqueeze(0) for item in data_list]
        tgt_seq = torch.cat(tgt_seq, dim=0)
        return [graph_data, tgt_seq]


if __name__ == '__main__':
    JobsDataset()
