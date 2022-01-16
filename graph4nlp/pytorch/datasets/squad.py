import nltk

from ..data.dataset import DoubleText2TextDataset


class SQuADDataset(DoubleText2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        if self.for_inference:
            return {"test": "test.txt"}
        else:
            return {"train": "train.txt", "val": "val.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 2 reserved keys should be fiiled: 'vocab' and 'data'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    def download(self):
        raise NotImplementedError(
            "This dataset is now under test and cannot be downloaded."
            "Please prepare the raw data yourself."
        )

    def __init__(
        self,
        root_dir,
        topology_subdir,
        graph_name,
        static_or_dynamic="static",
        topology_builder=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        share_vocab=True,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        max_word_vocab_size=None,
        min_word_vocab_freq=1,
        tokenizer=nltk.RegexpTokenizer(" ", gaps=True).tokenize,
        word_emb_size=None,
        **kwargs
    ):
        super(SQuADDataset, self).__init__(
            graph_name,
            root_dir=root_dir,
            static_or_dynamic=static_or_dynamic,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            share_vocab=share_vocab,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            max_word_vocab_size=max_word_vocab_size,
            min_word_vocab_freq=min_word_vocab_freq,
            tokenizer=tokenizer,
            word_emb_size=word_emb_size,
            **kwargs
        )
