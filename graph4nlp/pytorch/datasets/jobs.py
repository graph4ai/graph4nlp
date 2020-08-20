import os

import torch
import pickle


from graph4nlp.pytorch.data.dataset import Text2TextDataset, TextToTreeDataset
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction



dataset_root = '../test/dataset/jobs'

class JobsDataset(Text2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.txt', 'test': 'test.txt'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt', 'split_ids': 'split_ids.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', **kwargs):
        super(JobsDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy, **kwargs)
        self.data = torch.load(os.path.join(self.processed_dir, self.processed_file_names['data']))
        # self.build_vocab()
        self.vocab = pickle.load(open(os.path.join(self.processed_dir, self.processed_file_names['vocab']), 'rb'))


class JobsDatasetForTree(TextToTreeDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.txt', 'test': 'test.txt'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt', 'split_ids': 'split_ids.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', **kwargs):
        super(JobsDatasetForTree, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy, **kwargs)
        self.data = torch.load(os.path.join(self.processed_dir, self.processed_file_names['data']))
        self.split_ids = torch.load(os.path.join(self.processed_dir, self.processed_file_names['split_ids']))
        self.src_vocab_model, self.tgt_vocab_model = pickle.load(open(self.processed_file_paths['vocab'], 'rb'))

if __name__ == '__main__':
    a = JobsDatasetForTree(root_dir='/Users/lishucheng/Desktop/g4nlp/graph4nlp/graph4nlp/pytorch/test/dataset/jobs', topology_builder=ConstituencyBasedGraphConstruction,
                    topology_subdir='ConstiencyGraph', share_vocab=True, enc_emb_size=300, dec_emb_size=300)
    # print(len(a.split_ids["train"]))
    # print(a.split_ids['train'])
    # print(a.split_ids['test'])
    # print(a.src_vocab_model.symbol2idx)