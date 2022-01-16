from graph4nlp.pytorch.data.dataset import Text2TextDataset, Text2TreeDataset

from ..modules.graph_construction.dependency_graph_construction import (
    DependencyBasedGraphConstruction,
)

dataset_root = "../test/dataset/jobs"


class JobsDataset(Text2TextDataset):
    """

        Parameters
        ----------
        root_dir: str
            The path of dataset.
        graph_name: str
            The name of graph construction method. E.g., "dependency".
            Note that if it is in the provided graph names (i.e., "dependency", \
                "constituency", "ie", "node_emb", "node_emb_refine"), the following \
                parameters are set by default and users can't modify them:
                1. ``topology_builder``
                2. ``static_or_dynamic``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        topology_builder: GraphConstructionBase, default=None
            The graph construction class.
        topology_subdir: str
            The directory name of processed path.
        static_or_dynamic: str, default='static'
            The graph type. Expected in ('static', 'dynamic')
        edge_strategy: str, default=None
            The edge strategy. Expected in (None, 'homogeneous', 'as_node').
            If set `None`, it will be 'homogeneous'.
        merge_strategy: str, default=None
            The strategy to merge sub-graphs. Expected in (None, 'tailhead', 'user_define').
            If set `None`, it will be 'tailhead'.
        share_vocab: bool, default=False
            Whether to share the input vocabulary with the output vocabulary.
        dynamic_init_graph_name: str, default=None
            The graph name of the initial graph. Expected in (None, "line", "dependency", \
                "constituency").
            Note that if it is in the provided graph names (i.e., "line", "dependency", \
                "constituency"), the following parameters are set by default and users \
                can't modify them:
                1. ``dynamic_init_topology_builder``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        dynamic_init_topology_builder: GraphConstructionBase
            The graph construction class.
        dynamic_init_topology_aux_args: None,
            TBD.
        """

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
        topology_subdir,
        graph_name,
        static_or_dynamic="static",
        topology_builder=None,
        merge_strategy="tailhead",
        edge_strategy=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        pretrained_word_emb_name="6B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        seed=None,
        word_emb_size=300,
        share_vocab=True,
        lower_case=True,
        thread_number=1,
        port=9000,
        for_inference=None,
        reused_vocab_model=None,
    ):
        # Initialize the dataset. If the preprocessed files are not found,
        # then do the preprocessing and save them.
        super(JobsDataset, self).__init__(
            root_dir=root_dir,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_name=graph_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            share_vocab=share_vocab,
            lower_case=lower_case,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            seed=seed,
            word_emb_size=word_emb_size,
            thread_number=thread_number,
            port=port,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
        )


def tokenize_jobs(str_input):
    return str_input.strip().split()


class JobsDatasetForTree(Text2TreeDataset):
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
        tokenizer=tokenize_jobs,
        max_word_vocab_size=100000,
        for_inference=False,
        reused_vocab_model=None,
    ):
        """
        Parameters
        ----------
        root_dir: str
            The path of dataset.
        graph_name: str
            The name of graph construction method. E.g., "dependency".
            Note that if it is in the provided graph names (i.e., "dependency", \
                "constituency", "ie", "node_emb", "node_emb_refine"), the following \
                parameters are set by default and users can't modify them:
                1. ``topology_builder``
                2. ``static_or_dynamic``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        topology_builder: GraphConstructionBase
            The graph construction class.
        topology_subdir: str
            The directory name of processed path.
        static_or_dynamic: str, default='static'
            The graph type. Expected in ('static', 'dynamic')
        edge_strategy: str, default=None
            The edge strategy. Expected in (None, 'homogeneous', 'as_node').
            If set `None`, it will be 'homogeneous'.
        merge_strategy: str, default=None
            The strategy to merge sub-graphs. Expected in (None, 'tailhead', 'user_define').
            If set `None`, it will be 'tailhead'.
        share_vocab: bool, default=False
            Whether to share the input vocabulary with the output vocabulary.
        dynamic_init_graph_name: str, default=None
            The graph name of the initial graph. Expected in (None, "line", "dependency", \
                "constituency").
            Note that if it is in the provided graph names (i.e., "line", "dependency", \
                "constituency"), the following parameters are set by default and users \
                can't modify them:
                1. ``dynamic_init_topology_builder``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        dynamic_init_topology_builder: GraphConstructionBase
            The graph construction class.
        dynamic_init_topology_aux_args: None,
            TBD.
        """
        # Initialize the dataset. If the preprocessed files are not found,
        # then do the preprocessing and save them.
        super(JobsDatasetForTree, self).__init__(
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


if __name__ == "__main__":
    jobs_dataset = JobsDataset(
        root_dir="../test/dataset/jobs",
        topology_builder=DependencyBasedGraphConstruction,
        topology_subdir="DependencyGraph",
    )
    # Load train, val and test subsets
    train_size = len(jobs_dataset.split_ids["train"])
    # Since the validation file is not present in this example,
    # we use the 80% of the original training set as the
    # real training set and the rest 20% as the validation set
    train_set = jobs_dataset[jobs_dataset.split_ids["train"][: int(0.8 * train_size)]]
    val_set = jobs_dataset[jobs_dataset.split_ids["train"][int(0.8 * train_size) :]]

    test_set = jobs_dataset[jobs_dataset.split_ids["test"]]

    from torch.utils.data import dataloader

    train_dataloader = dataloader.DataLoader(
        dataset=train_set, batch_size=10, shuffle=True, collate_fn=jobs_dataset.collate_fn
    )
    print(
        "The number of batches in train_dataloader is {} with batch size of 10.".format(
            len(train_dataloader)
        )
    )

    # You can also use the built-in shuffle() method to obtain a shuffled
