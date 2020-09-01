import os

import torch
import pickle


from graph4nlp.pytorch.data.dataset import Text2TextDataset, TextToTreeDataset
from ..modules.graph_construction.base import GraphConstructionBase
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
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return

    def __init__(self, root_dir, topology_builder, topology_subdir, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', share_vocab=False, dynamic_graph_type=None,
                 init_graph_type=None):
        """

        Parameters
        ----------
        root_dir: str
            The path of dataset.
        topology_builder: GraphConstructionBase
            The graph construction class.
        topology_subdir: str
            The directory name of processed path.
        graph_type: str, default='static'
            The graph type. Expected in ('static', 'dynamic')
        edge_strategy: str, default=None
            The edge strategy. Expected in (None, 'homogeneous', 'as_node'). If set `None`, it will be 'homogeneous'.
        merge_strategy: str, default=None
            The strategy to merge sub-graphs. Expected in (None, 'tailhead', 'user_define').
            If set `None`, it will be 'tailhead'.
        share_vocab: bool, default=False
            Whether to share the input vocabulary with the output vocabulary.
        dynamic_graph_type: str, default=None
            The dynamic graph type. It is only available when `graph_type` is set 'dynamic'.
            Expected in (None, 'node_emb', 'node_emb_refined').
        init_graph_type: str, default=None
            The initial graph topology. It is only available when `graph_type` is set 'dynamic'.
            Expected in (None, 'dependency', 'constituency')
        """
        # Initialize the dataset. If the preprocessed files are not found, then do the preprocessing and save them.
        super(JobsDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy,
                                          share_vocab=share_vocab, dynamic_graph_type=dynamic_graph_type,
                                          init_graph_type=init_graph_type)

    @classmethod
    def from_args(cls, args, topology_builder):
        return cls(root_dir=args.root_dir, topology_builder=topology_builder, topology_subdir=args.topology_subdir,
                   graph_type=args.graph_type, edge_strategy=args.edge_strategy, merge_strategy=args.merge_strategy,
                   share_vocab=args.share_vocab, dynamic_graph_type=args.dynamic_graph_type,
                   init_graph_type=args.init_graph_type)


class JobsDatasetForTree(TextToTreeDataset):
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
        super(JobsDatasetForTree, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
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
