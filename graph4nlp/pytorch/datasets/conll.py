import os

import torch

from graph4nlp.pytorch.data.dataset import SequenceLabelingDataset
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction

dataset_root = '../test/dataset/conll/'


class ConllDataset(SequenceLabelingDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""        
        return {'train':'eng.train','test':'eng.testa','val':'eng.testb'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt', 'split_ids': 'split_ids.pt'}

    #def download(self):
    #    raise NotImplementedError(
    #        'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy=None,tag_types=None,**kwargs):
        super(ConllDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy,tag_types=tag_types, **kwargs)



if __name__ == '__main__':
    ConllDataset(root_dir='../test/dataset/conll/', topology_builder=DependencyBasedGraphConstruction,
                topology_subdir='DependencyGraph',tag_types=['I-MISC', 'O', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG'])

