from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
import torch
import os
import json
from stanfordcorenlp import StanfordCoreNLP
from multiprocessing import Pool
import numpy as np
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size

from multiprocessing import Process
import multiprocessing
import tqdm
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel, Vocab
from graph4nlp.pytorch.modules.utils import constants
from nltk.tokenize import word_tokenize

class CNNDataset(Text2TextDataset):
    def __init__(self,
                 root_dir,
                 topology_builder,
                 topology_subdir,
                 tokenizer=word_tokenize,
                 lower_case=True,
                 pretrained_word_emb_file=None,
                 use_val_for_vocab=False,
                 seed=1234,
                 device='cpu',
                 thread_number=4,
                 port=9000,
                 timeout=15000,
                 graph_type='static',
                 edge_strategy=None,
                 merge_strategy='tailhead',
                 share_vocab=True,
                 word_emb_size=300,
                 dynamic_graph_type=None,
                 dynamic_init_topology_builder=None,
                 dynamic_init_topology_aux_args=None
                 ):
        super(CNNDataset, self).__init__(root_dir=root_dir,
                                         topology_builder=topology_builder,
                                         topology_subdir=topology_subdir,
                                         tokenizer=tokenizer,
                                         lower_case=lower_case,
                                         pretrained_word_emb_file=pretrained_word_emb_file,
                                         use_val_for_vocab=use_val_for_vocab,
                                         seed=seed,
                                         device=device,
                                         thread_number=thread_number,
                                         port=port,
                                         timeout=timeout,
                                         graph_type=graph_type,
                                         edge_strategy=edge_strategy,
                                         merge_strategy=merge_strategy,
                                         share_vocab=share_vocab,
                                         word_emb_size=word_emb_size,
                                         dynamic_graph_type=dynamic_graph_type,
                                         dynamic_init_topology_builder=dynamic_init_topology_builder,
                                         dynamic_init_topology_aux_args=dynamic_init_topology_aux_args)

    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        # return {'train': 'train.json', 'val': "val.json", 'test': 'test.json'}
        # return {'train': 'train-0.json', 'val': "val-0.json", 'test': 'test-0.json'}
        return {'train': 'train_3w.json', 'val': "val.json", 'test': 'test.json'}
        # return {'train': 'train_9w.json', 'val': "val.json", 'test': 'test.json'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        return

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        vocab_model = VocabModel.build(saved_vocab_file=self.processed_file_paths['vocab'],
                                       data_set=data_for_vocab,
                                       tokenizer=self.tokenizer,
                                       lower_case=self.lower_case,
                                       max_word_vocab_size=None,
                                       min_word_vocab_freq=3,
                                       pretrained_word_emb_file=self.pretrained_word_emb_file,
                                       word_emb_size=self.word_emb_size,
                                       share_vocab=self.share_vocab)
        self.vocab_model = vocab_model
        return self.vocab_model

    def parse_file(self, file_path):
        """
        Read and parse the file specified by `file_path`. The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.
        For Text2TextDataset, the format of the input file should contain lines of input, each line representing one
        record of data. The input and output is separated by a tab(\t).
        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )
        DataItem: input_text="list job use languageid0", output_text="job ( ANS ) , language ( ANS , languageid0 )"
        Parameters
        ----------
        file_path: str
            The path of the input file.
        Returns
        -------
        list
            The indices of data items in the file w.r.t. the whole dataset.
        """
        data = []
        with open(file_path, 'r') as f:
            examples = json.load(f)
            for example_dict in examples:
                # input = ' '.join(example_dict['article'][:10]).lower()
                # output = ' '.join([sent[0]+' .' for sent in example_dict['highlight']]).lower()
                # input = ' '.join(' '.join(example_dict['article']).split()).lower()
                # input = input+input+input+input+input
                input = ' '.join(' '.join(example_dict['article']).split()[:500]).lower()
                output = ' '.join(' '.join(['<t> ' + sent[0] + ' . </t>' for sent in example_dict['highlight']]).split()[:99]).lower()
                if input=='' or output=='':
                    continue
                # output = ' '.join(["%s %s %s" % (constants._SOS_TOKEN, sent[0], constants._EOS_TOKEN) for sent in example_dict['highlight']])
                data_item = Text2TextDataItem(input_text=input, output_text=output, tokenizer=self.tokenizer,
                                              share_vocab=self.share_vocab)
                data.append(data_item)
        return data


if __name__ == "__main__":
    dataset = CNNDataset(root_dir="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn",
                         topology_builder=DependencyBasedGraphConstruction,
                         topology_subdir='DependencyGraph_3w',
                         word_emb_size=128,
                         share_vocab=True)

    # dataset = CNNDataset(root_dir="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn",
    #                      topology_builder=LinearGraphConstruction,
    #                      topology_subdir='LinearGraph',
    #                      word_emb_size=128,
    #                      share_vocab=True)

    # dataset = CNNSeq2SeqDataset(root_dir="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn",
    #                             topology_builder=DependencyBasedGraphConstruction,
    #                             topology_subdir='DependencyGraph_seq2seq', share_vocab=True)
    # a  = 0