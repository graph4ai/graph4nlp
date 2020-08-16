import os

import torch

from graph4nlp.pytorch.data.dataset import Text2TextDataset
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction

dataset_root = '../test/dataset/jobs'


class JobsDataset(Text2TextDataset):
    @property
    def raw_file_names(self):
        return {'train': 'train.txt', 'test': 'test.txt'}

    @property
    def processed_file_names(self):
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        raise NotImplementedError(
            'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', **kwargs):
        super(JobsDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy, **kwargs)
        self.data = torch.load(os.path.join(self.processed_dir, self.processed_file_names['data']))
        self.build_vocab()


if __name__ == '__main__':
    JobsDataset(root_dir='../test/dataset/jobs', topology_builder=DependencyBasedGraphConstruction,
                topology_subdir='DependencyGraph')
