import os

import torch
import pickle


from graph4nlp.pytorch.data.dataset import TextToTreeDataset
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction

class GeoDatasetForTree(TextToTreeDataset):
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
        super(GeoDatasetForTree, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy,
                                          share_vocab=share_vocab, pretrained_word_emb_file=pretrained_word_emb_file,
                                          val_split_ratio=val_split_ratio, seed=seed, word_emb_size=word_emb_size,

                                          dynamic_graph_type=dynamic_graph_type,
                                          dynamic_init_topology_builder=dynamic_init_topology_builder,
                                          dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
                                          enc_emb_size=enc_emb_size, dec_emb_size=dec_emb_size, device=device,
                                          min_freq=min_freq)

