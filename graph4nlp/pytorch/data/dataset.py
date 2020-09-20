import abc
import os

import numpy as np
import random
import stanfordcorenlp
import torch.utils.data
from nltk.tokenize import word_tokenize
from sklearn import preprocessing

from collections import Counter
import pickle

from ..data.data import GraphData
from ..modules.utils.vocab_utils import VocabModel, Vocab

from ..modules.utils.tree_utils import Vocab as VocabForTree
from ..modules.utils.tree_utils import Tree

import json
from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from ..modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
from ..modules.graph_construction.dependency_graph_construction_without_tokenize import DependencyBasedGraphConstruction_without_tokenizer
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size


class DataItem(object):
    def __init__(self, input_text, tokenizer):
        self.input_text = input_text
        self.tokenizer = tokenizer
        pass

    @abc.abstractmethod
    def extract(self):
        raise NotImplementedError


class Text2TextDataItem(DataItem):
    def __init__(self, input_text, output_text, tokenizer, share_vocab=True):
        super(Text2TextDataItem, self).__init__(input_text, tokenizer)
        self.output_text = output_text
        self.share_vocab = share_vocab

    def extract(self):
        """
        Returns
        -------
        Input tokens and output tokens
        """
        g: GraphData = self.graph

        input_tokens = []
        for i in range(g.get_node_num()):
            if self.tokenizer is None:
                tokenized_token = self.output_text.strip().split(' ')
            else:
                tokenized_token = self.tokenizer(g.node_attributes[i]['token'])

            input_tokens.extend(tokenized_token)

        if self.tokenizer is None:
            output_tokens = self.output_text.strip().split(' ')
        else:
            output_tokens = self.tokenizer(self.output_text)

        if self.share_vocab:
            return input_tokens + output_tokens
        else:
            return input_tokens, output_tokens


class SequenceLabelingDataItem(DataItem):
    def __init__(self, input_text, output_tags, tokenizer):
        super(SequenceLabelingDataItem, self).__init__(input_text, tokenizer)
        self.output_tag = output_tags

    def extract(self):
        """
        Returns
        -------
        Input tokens and output tags
        """
        g: GraphData = self.graph

        input_tokens = []
        for i in range(g.get_node_num()):
            if self.tokenizer is None:
                tokenized_token = self.output_text.strip().split(' ')
            else:
                tokenized_token = self.tokenizer(g.node_attributes[i]['token'])

            input_tokens.extend(tokenized_token)


        return input_tokens 

class Dataset(torch.utils.data.Dataset):
    """
    Base class for datasets.
    The dataset is organized in a two-layer index style. Direct access to the dataset object, e.g. Dataset[1], will first
    be converted to the access to the internal index list, which is then passed to access the actual data. This design
    is for the ease of sampling.
    
    Examples
    --------
    Suppose we have a Dataset containing 5 data items ['a', 'b', 'c', 'd', 'e']. The indices of the 5 elements in the
    list are correspondingly [0, 1, 2, 3, 4]. Suppose the dataset is shuffled, which shuffles the internal index list, the
    consequent indices becomes [2, 3, 1, 4, 5]. Then an access to the dataset `Dataset[2]` will first access the indices[2]
    which is 1, and then use the received index to access the actual dataset, which will return the actual data item 'b'.
    Now to the user the 3rd ([2]) element in the dataset got shuffled and is not 'c'.
    
    Parameters
    ----------
    root: str
        The root directory path where the dataset is stored.
    """

    @property
    def raw_file_names(self) -> dict:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def download(self):
        """Download the raw data from the Internet."""
        raise NotImplementedError

    @abc.abstractmethod
    def vectorization(self, data_items):
        """Convert tokens to indices which can be processed by downstream models."""
        raise NotImplementedError

    @abc.abstractmethod
    def parse_file(self, file_path):
        """To be implemented in task-specific dataset base class."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def collate_fn(data_list):
        """Takes a list of data and convert it to a batch of data."""
        raise NotImplementedError

    def __init__(self,
                 root,
                 topology_builder,
                 topology_subdir,
                 tokenizer=word_tokenize,
                 lower_case=True,
                 pretrained_word_emb_file=None,
                 use_val_for_vocab=False,
                 **kwargs):
        super(Dataset, self).__init__()

        self.root = root    # The root directory where the dataset is located.

        # Processing-specific attributes
        self.tokenizer = tokenizer
        self.lower_case = lower_case
        self.pretrained_word_emb_file = pretrained_word_emb_file
        self.topology_builder = topology_builder
        self.topology_subdir = topology_subdir
        self.use_val_for_vocab = use_val_for_vocab
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__indices__ = None

        if 'download' in self.__class__.__dict__.keys():
            self._download()
            
        self._process()         
        
        # After initialization, load the preprocessed files.
        data = torch.load(self.processed_file_paths['data'])
        self.train = data['train']
        self.test = data['test']
        if 'val' in data.keys():
            self.val = data['val']            

        self.build_vocab()


    @property
    def raw_dir(self) -> str:
        """The directory where the raw data is stored."""
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed', self.topology_subdir)

    @property
    def raw_file_paths(self) -> dict:
        """The paths to raw files."""
        return {key: os.path.join(self.raw_dir, name) for key, name in self.raw_file_names.items()}

    @property
    def processed_file_paths(self) -> dict:
        return {name: os.path.join(self.processed_dir, processed_file_name) for name, processed_file_name in
                self.processed_file_names.items()}

    def _download(self):
        if all([os.path.exists(raw_path) for raw_path in self.raw_file_paths.values()]):
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def read_raw_data(self):
        """
        Read raw data from the disk and put them in a dictionary (`self.data`).
        The raw data file should be organized as the format defined in `self.parse_file()` method.
        This function calls `self.parse_file()` repeatedly and pass the file paths in `self.raw_file_names` once at a time.
        This function builds `self.data` which is a dict of {int (index): DataItem}, where the id represents the
        index of the DataItem w.r.t. the whole dataset.
        This function also builds the `self.split_ids` dictionary whose keys correspond to those of self.raw_file_names
        defined by the user, indicating the indices of each subset (e.g. train, val and test).
        """
        self.train = self.parse_file(self.raw_file_paths['train'])
        self.test = self.parse_file(self.raw_file_paths['test'])
        if 'val' in self.raw_file_paths.keys():
            self.val = self.parse_file(self.raw_file_paths['val'])
        elif 'val_split_ratio' in self.__dict__:
            if self.val_split_ratio > 0:
                new_train_length = int((1 - self.val_split_ratio) * len(self.train))
                import random
                old_train_set = self.train
                random.shuffle(old_train_set)
                self.val = old_train_set[new_train_length:]
                self.train = old_train_set[:new_train_length]

    def build_topology(self, data_items):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        if self.graph_type == 'static':
            print('Connecting to stanfordcorenlp server...')
            processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)

            if self.topology_builder == IEBasedGraphConstruction:
                props_coref = {
                    'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
                    "tokenize.options":
                        "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
                props_openie = {
                    'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
                    "tokenize.options":
                        "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json',
                    "openie.triple.strict": "true"
                }
                processor_args = [props_coref, props_openie]
            elif self.topology_builder == DependencyBasedGraphConstruction:
                processor_args = {
                    'annotators': 'ssplit,tokenize,depparse',
                    "tokenize.options":
                        "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
            elif self.topology_builder == ConstituencyBasedGraphConstruction:
                processor_args = {
                    'annotators': "tokenize,ssplit,pos,parse",
                    "tokenize.options":
                    "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
            else:
                raise NotImplementedError
            print('CoreNLP server connected.')
            for item in data_items:
                graph = self.topology_builder.topology(raw_text_data=item.input_text,
                                                       nlp_processor=processor,
                                                       processor_args=processor_args,
                                                       merge_strategy=self.merge_strategy,
                                                       edge_strategy=self.edge_strategy,
                                                       verbase=False)
                item.graph = graph
        elif self.graph_type == 'dynamic':
            if self.dynamic_graph_type == 'node_emb':
                for item in data_items:
                    graph = self.topology_builder.init_topology(item.input_text,
                                                                lower_case=self.lower_case,
                                                                tokenizer=self.tokenizer)
                    item.graph = graph
            elif self.dynamic_graph_type == 'node_emb_refined':
                if self.dynamic_init_topology_builder in (IEBasedGraphConstruction, DependencyBasedGraphConstruction, ConstituencyBasedGraphConstruction):
                    print('Connecting to stanfordcorenlp server...')
                    processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)

                    if self.dynamic_init_topology_builder == IEBasedGraphConstruction:
                        props_coref = {
                            'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
                            "tokenize.options":
                                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                        props_openie = {
                            'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
                            "tokenize.options":
                                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json',
                            "openie.triple.strict": "true"
                        }
                        processor_args = [props_coref, props_openie]
                    elif self.dynamic_init_topology_builder == DependencyBasedGraphConstruction:
                        processor_args = {
                            'annotators': 'ssplit,tokenize,depparse',
                            "tokenize.options":
                                "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                    elif self.dynamic_init_topology_builder == ConstituencyBasedGraphConstruction:
                        processor_args = {
                            'annotators': "tokenize,ssplit,pos,parse",
                            "tokenize.options":
                            "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                    else:
                        raise NotImplementedError
                    print('CoreNLP server connected.')
                else:
                    processor = None
                    processor_args = None

                for item in data_items:
                    graph = self.topology_builder.init_topology(item.input_text,
                                                                dynamic_init_topology_builder=self.dynamic_init_topology_builder,
                                                                lower_case=self.lower_case,
                                                                tokenizer=self.tokenizer,
                                                                nlp_processor=processor,
                                                                processor_args=processor_args,
                                                                merge_strategy=self.merge_strategy,
                                                                edge_strategy=self.edge_strategy,
                                                                verbase=False,
                                                                dynamic_init_topology_aux_args=self.dynamic_init_topology_aux_args)

                    item.graph = graph
            else:
                raise RuntimeError('Unknown dynamic_graph_type: {}'.format(self.dynamic_graph_type))

        else:
            raise NotImplementedError('Currently only static and dynamic are supported!')

    def build_vocab(self):
        """
        Build the vocabulary. If `self.use_val_for_vocab` is `True`, use both training set and validation set for building
        the vocabulary. Otherwise only the training set is used.
        """
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = self.val + data_for_vocab

        vocab_model = VocabModel.build(saved_vocab_file=self.processed_file_paths['vocab'],
                                       data_set=data_for_vocab,
                                       tokenizer=self.tokenizer,
                                       lower_case=self.lower_case,
                                       max_word_vocab_size=self.max_word_vocab_size,
                                       min_word_vocab_freq=self.min_word_vocab_size,
                                       pretrained_word_emb_file=self.pretrained_word_emb_file,
                                       word_emb_size=self.word_emb_size)
        self.vocab_model = vocab_model

        return self.vocab_model

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths.values()]):
            if 'val_split_ratio' in self.__dict__:
                UserWarning(
                    "Loading existing processed files on disk. Your `val_split_ratio` might not work since the data have"
                    "already been split.")
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        self.read_raw_data()

        self.build_topology(self.train)
        self.build_topology(self.test)
        if 'val' in self.__dict__:
            self.build_topology(self.val)

        self.build_vocab()

        self.vectorization(self.train)
        self.vectorization(self.test)
        if 'val' in self.__dict__:
            self.vectorization(self.val)

        data_to_save = {'train': self.train, 'test': self.test}
        if 'val' in self.__dict__:
            data_to_save['val'] = self.val
        torch.save(data_to_save, self.processed_file_paths['data'])


class Text2TextDataset(Dataset):
    def __init__(self, root_dir, topology_builder, topology_subdir, share_vocab=True, **kwargs):
        self.data_item_type = Text2TextDataItem
        self.share_vocab = share_vocab
        super(Text2TextDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def parse_file(self, file_path) -> list:
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
            lines = f.readlines()
            for line in lines:
                input, output = line.split('\t')
                data_item = Text2TextDataItem(input_text=input, output_text=output, tokenizer=self.tokenizer,
                                              share_vocab=self.share_vocab)
                data.append(data_item)
        return data

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        vocab_model = VocabModel.build(saved_vocab_file=self.processed_file_paths['vocab'],
                                       data_set=data_for_vocab,
                                       tokenizer=self.tokenizer,
                                       lower_case=self.lower_case,
                                       max_word_vocab_size=None,
                                       min_word_vocab_freq=1,
                                       pretrained_word_emb_file=self.pretrained_word_emb_file,
                                       word_emb_size=300,
                                       share_vocab=self.share_vocab)
        self.vocab_model = vocab_model

        return self.vocab_model

    def vectorization(self, data_items):
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features['token_id'] = token_matrix

            tgt = item.output_text
            tgt_token_id = self.vocab_model.in_word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(self.vocab_model.in_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            tgt_token_id = torch.from_numpy(tgt_token_id)
            item.output_tensor = tgt_token_id

    @staticmethod
    def collate_fn(data_list: [Text2TextDataItem]):
        graph_data = [item.graph for item in data_list]

        # do padding here
        seq_len = [item.output_tensor.shape[0] for item in data_list]
        max_seq_len = max(seq_len)
        tgt_seq_pad = []
        for item in data_list:
            if item.output_tensor.shape[0] < max_seq_len:
                need_pad_length = max_seq_len - item.output_tensor.shape[0]
                pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
                tgt_seq_pad.append(torch.cat((item.output_tensor, pad.long()), dim=0).unsqueeze(0))
            elif item.output_tensor.shape[0] == max_seq_len:
                tgt_seq_pad.append(item.output_tensor.unsqueeze(0))
            else:
                raise RuntimeError("Size mismatch error")

        tgt_seq = torch.cat(tgt_seq_pad, dim=0)
        return [graph_data, tgt_seq]


class SequenceLabelingDataset(Dataset):
    def __init__(self, root_dir, topology_builder, topology_subdir,tag_types, **kwargs):
        self.data_item_type = SequenceLabelingDataItem
        self.tag_types=tag_types
        super(SequenceLabelingDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.
        For SequenceLabelingDataset, the format of the input file should contain lines of tokens, each line representing one
        record of token at first column and its tag at the last column. 
        Examples
        --------
        "EU       I-ORG "
         rejects  O
         German   I-MISC
         
        Parameters
        ----------

        """
        data=[]
        input=[]
        output=[]
        with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if len(line)>1 and line[0]!='-':
                       if line[0]!='.': 
                           input.append(line.strip().split(' ')[0])
                           output.append(line.strip().split(' ')[-1])                        
                       if line[0]=='.':
                           input.append(line.strip().split(' ')[0])
                           output.append(line.strip().split(' ')[-1])
                           if len(input) >= 2:
                               data_item=SequenceLabelingDataItem(input_text=input, output_tags=output, tokenizer=self.tokenizer)
                               data.append(data_item)
                               input=[]
                               output=[]                                                

        return data     


    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val
            
        vocab_model = VocabModel.build(saved_vocab_file=self.processed_file_paths['vocab'],
                                       data_set=data_for_vocab,
                                       tokenizer=self.tokenizer,
                                       lower_case=self.lower_case,
                                       max_word_vocab_size=None,
                                       min_word_vocab_freq=1,
                                       pretrained_word_emb_file=self.pretrained_word_emb_file,
                                       word_emb_size=300,
                                       share_vocab=True)
        self.vocab_model = vocab_model

        return self.vocab_model

    def vectorization(self, data_items):
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features['token_id'] = token_matrix

            tgt = item.output_tag
            tgt_tag_id=[self.tag_types.index(tgt_.strip()) for tgt_ in tgt]
            
            tgt_tag_id = torch.tensor(tgt_tag_id)
            item.output_id = tgt_tag_id

    @staticmethod
    def collate_fn(data_list: [SequenceLabelingDataItem]):
        tgt_tag=[]
        graph_data=[]
        for item in data_list:
           #if len(item.graph.node_attributes)== len(item.output_id):
                   graph_data.append(item.graph)
                   tgt_tag.append(item.output_id)

        #tgt_tags = torch.cat(tgt_tag, dim=0)
        return [graph_data, tgt_tag]











