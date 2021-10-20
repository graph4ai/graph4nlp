from ..data.dataset import Text2LabelDataset


class TrecDataset(Text2LabelDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "train.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'label'."""
        return {"vocab": "vocab.pt", "data": "data.pt", "label": "label.pt"}

    def download(self):
        # raise NotImplementedError(
        #     "This dataset is now under test and cannot be downloaded."
        #     "Please prepare the raw data yourself."
        #     )
        return

    def __init__(
        self,
        root_dir,
        topology_subdir=None,
        graph_type="none",
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        edge_strategy=None,
        max_word_vocab_size=None,
        min_word_vocab_freq=1,
        word_emb_size=None,
        for_inference=None,
        reused_vocab_model=None,
        reused_label_model=None,
        dynamic_init_graph_type=None,
        **kwargs
    ):
        super(TrecDataset, self).__init__(
            root_dir=root_dir,
            topology_subdir=topology_subdir,
            graph_type=graph_type,
            edge_strategy=edge_strategy,
            max_word_vocab_size=max_word_vocab_size,
            min_word_vocab_freq=min_word_vocab_freq,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            word_emb_size=word_emb_size,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
            reused_label_model=reused_label_model,
            dynamic_init_graph_type=dynamic_init_graph_type,
            **kwargs
        )
