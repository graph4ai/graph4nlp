import os
import torch
import json
from nltk.tokenize import word_tokenize
from graph4nlp.pytorch.data.dataset import KGCompletionDataset, KGDataItem
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from ..data.data import GraphData
from ..modules.utils.vocab_utils import Vocab
import numpy as np

dataset_root = '../test/dataset/kinship'


class KinshipDataset(KGCompletionDataset):
    @property
    def raw_file_names(self) -> dict:
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'e1rel_to_e2_ranking_train.json',
                'val': 'e1rel_to_e2_ranking_dev.json',
                'test': 'e1rel_to_e2_ranking_test.json'}

    @property
    def processed_file_names(self) -> dict:
        return {'vocab': 'vocab.pt', 'data': 'data.pt', 'KG_graph': 'KG_graph.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, edge_strategy=None, **kwargs):
        self.split_token = ' '
        super(KinshipDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                             topology_subdir=topology_subdir,edge_strategy=edge_strategy, **kwargs)


if __name__ == '__main__':
    kinshipdataset = KinshipDataset(root_dir='/Users/gaohanning/PycharmProjects/graph4nlp/examples/pytorch/kg_completion/kinship',
                                    topology_builder=None,
                                    topology_subdir='e1rel_to_e2')

    a = 0
