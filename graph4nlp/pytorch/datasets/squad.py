import os
import torch
import nltk

from ..data.dataset import DoubleText2TextDataset


class SQuADDataset(DoubleText2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}

    @property
    def processed_file_names(self):
        """At least 2 reserved keys should be fiiled: 'vocab' and 'data'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        raise NotImplementedError(
            'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 pretrained_word_emb_name="840B", pretrained_word_emb_url=None,
                 edge_strategy=None, merge_strategy='tailhead', max_word_vocab_size=None,
                 tokenizer=nltk.RegexpTokenizer(" ", gaps=True).tokenize,
                 min_word_vocab_freq=1, word_emb_size=None, **kwargs):
        super(SQuADDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy,
                                          max_word_vocab_size=max_word_vocab_size,
                                          min_word_vocab_freq=min_word_vocab_freq,
                                          pretrained_word_emb_name=pretrained_word_emb_name, pretrained_word_emb_url=pretrained_word_emb_url,
                                          tokenizer=tokenizer,
                                          word_emb_size=word_emb_size, **kwargs)
