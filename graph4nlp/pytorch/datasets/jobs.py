import torch
import numpy as np


from graph4nlp.pytorch.data.dataset import Text2TextDataset, TextToTreeDataset
from ..modules.graph_construction.base import GraphConstructionBase
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..data.data import GraphData


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

    def __init__(self, root_dir,
                 topology_builder, topology_subdir,
                 pretrained_word_emb_file=None,
                 val_split_ratio=None,
                 graph_type='static',
                 merge_strategy="tailhead", edge_strategy=None,
                 seed=None,
                 word_emb_size=300, share_vocab=True,
                 dynamic_graph_type=None,
                 dynamic_init_topology_builder=None,
                 dynamic_init_topology_aux_args=None):
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
                                          share_vocab=share_vocab, pretrained_word_emb_file=pretrained_word_emb_file,
                                          val_split_ratio=val_split_ratio, seed=seed, word_emb_size=word_emb_size,

                                          dynamic_graph_type=dynamic_graph_type,
                                          dynamic_init_topology_builder=dynamic_init_topology_builder,
                                          dynamic_init_topology_aux_args=dynamic_init_topology_aux_args)

    # @classmethod
    # def from_args(cls, args, graph_type, topology_builder, dynamic_graph_type=None, dynamic_init_topology_builder=None,
    #               dynamic_init_topology_aux_args=None):
    #     return cls(root_dir=args.root_dir,
    #                pretrained_word_emb_file=args.pretrained_word_emb_file,
    #                val_split_ratio=args.val_split_ratio,
    #                merge_strategy=args.merge_strategy, edge_strategy=args.edge_strategy,
    #                seed=args.seed,
    #                word_emb_size=args.word_emb_size, share_vocab=args.share_vocab,
    #                graph_type=graph_type,
    #                topology_builder=topology_builder, topology_subdir=args.topology_subdir,
    #                dynamic_graph_type= dynamic_graph_type,
    #                dynamic_init_topology_builder=dynamic_init_topology_builder,
    #                dynamic_init_topology_aux_args=dynamic_init_topology_aux_args)




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

    def __init__(self, root_dir,
                 topology_builder, topology_subdir,
                 pretrained_word_emb_file=None,
                 val_split_ratio=0,
                 graph_type='static',
                 merge_strategy="tailhead", edge_strategy=None,
                 seed=None,
                 word_emb_size=300, share_vocab=True,
                 dynamic_graph_type=None,
                 dynamic_init_topology_builder=None,
                 dynamic_init_topology_aux_args=None,
                 enc_emb_size=300,
                 dec_emb_size=300,
                 device='cpu',
                 min_freq=1):
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
        super(JobsDatasetForTree, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy,
                                          share_vocab=share_vocab, pretrained_word_emb_file=pretrained_word_emb_file,
                                          val_split_ratio=val_split_ratio, seed=seed, word_emb_size=word_emb_size,

                                          dynamic_graph_type=dynamic_graph_type,
                                          dynamic_init_topology_builder=dynamic_init_topology_builder,
                                          dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
                                          enc_emb_size=enc_emb_size, dec_emb_size=dec_emb_size, device=device,
                                          min_freq=min_freq)


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