import os

import torch
import pickle

from graph4nlp.pytorch.data.dataset import Text2TextDataset
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction

dataset_root = '../test/dataset/jobs'


class JobsDataset(Text2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.txt', 'test': 'test.txt'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', **kwargs):
        # Initialize the dataset. If the preprocessed files are not found, then do the preprocessing and save them.
        super(JobsDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy, **kwargs)


if __name__ == '__main__':
    jobs_dataset = JobsDataset(root_dir='../test/dataset/jobs', topology_builder=DependencyBasedGraphConstruction,
                               topology_subdir='DependencyGraph')
    # Load train, val and test subsets
    train_size = len(jobs_dataset.split_ids['train'])
    # Since the validation file is not present in this example, we use the 80% of the original training set as the
    # real training set and the rest 20% as the validation set
    train_set = jobs_dataset[jobs_dataset.split_ids['train'][:int(0.8 * train_size)]]
    val_set = jobs_dataset[jobs_dataset.split_ids['train'][int(0.8 * train_size):]]

    test_set = jobs_dataset[jobs_dataset.split_ids['test']]

    from torch.utils.data import dataloader

    train_dataloader = dataloader.DataLoader(dataset=train_set, batch_size=10, shuffle=True,
                                             collate_fn=jobs_dataset.collate_fn)
    print('The number of batches in train_dataloader is {} with batch size of 10.'.format(len(train_dataloader)))

    # You can also use the built-in shuffle() method to obtain a shuffled
