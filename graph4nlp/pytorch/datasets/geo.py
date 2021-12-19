from graph4nlp.pytorch.data.dataset import Text2TreeDataset


def tokenize_geo(str_input):
    return str_input.strip().split()


class GeoDatasetForTree(Text2TreeDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "train.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded.
        # Please prepare the raw data yourself.')
        return

    def __init__(
        self,
        root_dir,
        # topology_builder,
        topology_subdir,
        graph_name,
        static_or_dynamic="static",
        topology_builder=None,
        merge_strategy="tailhead",
        edge_strategy=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        #  pretrained_word_emb_file=None,
        pretrained_word_emb_name="6B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        val_split_ratio=0,
        word_emb_size=300,
        share_vocab=True,
        enc_emb_size=300,
        dec_emb_size=300,
        min_word_vocab_freq=1,
        tokenizer=tokenize_geo,
        max_word_vocab_size=100000,
        for_inference=False,
        reused_vocab_model=None,
    ):
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
            The edge strategy. Expected in (None, 'homogeneous', 'as_node').
            If set `None`, it will be 'homogeneous'.
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
        # Initialize the dataset. If the preprocessed files are not found,
        # then do the preprocessing and save them.
        super(GeoDatasetForTree, self).__init__(
            root_dir=root_dir,
            # topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_name=graph_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            share_vocab=share_vocab,
            pretrained_word_emb_name=pretrained_word_emb_name,
            val_split_ratio=val_split_ratio,
            word_emb_size=word_emb_size,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            enc_emb_size=enc_emb_size,
            dec_emb_size=dec_emb_size,
            min_word_vocab_freq=min_word_vocab_freq,
            tokenizer=tokenizer,
            max_word_vocab_size=max_word_vocab_size,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
        )
