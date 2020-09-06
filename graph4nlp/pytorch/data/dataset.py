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
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
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
                tokenized_token = g.node_attributes[i]['token'].strip().split(' ')
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


class Text2TreeDataItem(DataItem):
    def __init__(self, input_text, output_text, output_tree, tokenizer, share_vocab=True):
        super(Text2TreeDataItem, self).__init__(input_text, tokenizer)
        self.output_text = output_text
        self.share_vocab = share_vocab
        self.output_tree = output_tree

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


class Text2LabelDataItem(DataItem):
    def __init__(self, input_text, output_label, tokenizer):
        super(Text2LabelDataItem, self).__init__(input_text, tokenizer)
        self.output_label = output_label

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
                tokenized_token = g.node_attributes[i]['token'].strip().split()
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
                 seed=1234,
                 **kwargs):
        super(Dataset, self).__init__()

        self.root = root  # The root directory where the dataset is located.
        self.seed = seed

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

        if 'KG_graph' in self.processed_file_paths.keys():
            self.KG_graph = torch.load(self.processed_file_paths['KG_graph'])

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
                random.seed(self.seed)
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
                if self.init_graph_type != 'line':
                    print('Connecting to stanfordcorenlp server...')
                    processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)

                    if self.init_graph_type == 'ie':
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
                    elif self.init_graph_type == 'dependency':
                        processor_args = {
                            'annotators': 'ssplit,tokenize,depparse',
                            "tokenize.options":
                                "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                    elif self.init_graph_type == 'constituency':
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
                                                                init_graph_type=self.init_graph_type,
                                                                lower_case=self.lower_case,
                                                                tokenizer=self.tokenizer,
                                                                nlp_processor=processor,
                                                                processor_args=processor_args,
                                                                merge_strategy=self.merge_strategy,
                                                                edge_strategy=self.edge_strategy,
                                                                verbase=False)
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
        if self.topology_builder == IEBasedGraphConstruction:
            use_ie = True
        else:
            use_ie = False
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token, use_ie)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            if self.topology_builder == IEBasedGraphConstruction:
                for i in range(len(token_matrix)):
                    token_matrix[i] = np.array(token_matrix[i][0])
                token_matrix = pad_2d_vals_no_size(token_matrix)
                token_matrix = torch.tensor(token_matrix, dtype=torch.long)
                graph.node_features['token_id'] = token_matrix
                pass
            else:
                token_matrix = torch.tensor(token_matrix, dtype=torch.long)
                graph.node_features['token_id'] = token_matrix

            if use_ie and 'token' in graph.edge_attributes[0].keys():
                edge_token_matrix = []
                for edge_idx in range(graph.get_edge_num()):
                    edge_token = graph.edge_attributes[edge_idx]['token']
                    edge_token_id = self.vocab_model.in_word_vocab.getIndex(edge_token, use_ie)
                    graph.edge_attributes[edge_idx]['token_id'] = edge_token_id
                    edge_token_matrix.append([edge_token_id])
                if self.topology_builder == IEBasedGraphConstruction:
                    for i in range(len(edge_token_matrix)):
                        edge_token_matrix[i] = np.array(edge_token_matrix[i][0])
                    edge_token_matrix = pad_2d_vals_no_size(edge_token_matrix)
                    edge_token_matrix = torch.tensor(edge_token_matrix, dtype=torch.long)
                    graph.edge_features['token_id'] = edge_token_matrix

            tgt = item.output_text
            tgt_token_id = self.vocab_model.in_word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(self.vocab_model.in_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            item.output_np = tgt_token_id

    @staticmethod
    def collate_fn(data_list: [Text2TextDataItem]):
        graph_data = [item.graph for item in data_list]

        output_numpy = [item.output_np for item in data_list]
        output_pad = pad_2d_vals_no_size(output_numpy)

        tgt_seq = torch.from_numpy(output_pad).long()
        return [graph_data, tgt_seq]


class TextToTreeDataset(Dataset):
    def __init__(self, root_dir, topology_builder, topology_subdir, share_vocab=True, **kwargs):
        self.data_item_type = Text2TreeDataItem
        self.share_vocab = share_vocab
        super(TextToTreeDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.

        For Text2TreeDataset, the format of the input file should contain lines of input, each line representing one
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
                data_item = Text2TreeDataItem(input_text=input, output_text=output, output_tree=None, tokenizer=self.tokenizer,
                                              share_vocab=self.share_vocab)
                data.append(data_item)
        return data

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        src_vocab_model = VocabForTree(lower_case=self.lower_case, pretrained_embedding_fn=self.pretrained_word_emb_file, embedding_dims=self.enc_emb_size)
        tgt_vocab_model = VocabForTree(lower_case=self.lower_case, pretrained_embedding_fn=self.pretrained_word_emb_file, embedding_dims=self.dec_emb_size)

        if self.share_vocab:
            all_words = Counter()
        else:
            all_words = [Counter(), Counter()]

        for instance in data_for_vocab:
            extracted_tokens = instance.extract()
            if self.share_vocab:
                all_words.update(extracted_tokens)
            else:
                all_words[0].update(extracted_tokens[0])
                all_words[1].update(extracted_tokens[1])

        if self.share_vocab:
            src_vocab_model.init_from_list(list(all_words.items()), min_freq=1, max_vocab_size=100000)
            tgt_vocab_model = src_vocab_model
        else:
            src_vocab_model.init_from_list(list(all_words[0].items()), min_freq=2, max_vocab_size=100000)
            tgt_vocab_model.init_from_list(list(all_words[1].items()), min_freq=1, max_vocab_size=100000)

        self.src_vocab_model = src_vocab_model
        self.tgt_vocab_model = tgt_vocab_model
        return self.src_vocab_model

    def vectorization(self, data_items):
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.src_vocab_model.get_symbol_idx(node_token)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features['token_id'] = token_matrix

            tgt = item.output_text
            tgt_list = self.tgt_vocab_model.get_symbol_idx_for_list(tgt.split())
            output_tree = Tree.convert_to_tree(tgt_list, 0, len(tgt_list), self.tgt_vocab_model)
            item.output_tree = output_tree

    @staticmethod
    def collate_fn(data_list: [Text2TreeDataItem]):
        graph_data = [item.graph for item in data_list]
        output_tree_list = [item.output_tree for item in data_list]
        return [graph_data, output_tree_list]

class KGDataItem(DataItem):
    def __init__(self, e1, rel, e2, rel_eval, e2_multi, e1_multi, share_vocab=True, split_token=' '):
        super(KGDataItem, self).__init__(input_text=None, tokenizer=None)
        self.e1 = e1
        self.rel = rel
        self.e2 = e2
        self.rel_eval = rel_eval
        self.e2_multi = e2_multi
        self.e1_multi = e1_multi
        self.share_vocab = share_vocab
        self.split_token = split_token

    def extract(self):
        """
        Returns
        -------
        Input tokens and output tokens
        """
        # g: GraphData = self.graph

        input_tokens = []
        if self.tokenizer is None:
            e1_tokens = self.e1.strip().split(' ')
        else:
            e1_tokens = self.tokenizer(self.e1)

        if self.tokenizer is None:
            e2_tokens = self.e2.strip().split(' ')
        else:
            e2_tokens = self.tokenizer(self.e2)

        if self.tokenizer is None:
            e2_multi_tokens = self.e2_multi.strip().split(' ')
        else:
            e2_multi_tokens = self.tokenizer(self.e2_multi)

        if self.tokenizer is None:
            e1_multi_tokens = self.e1_multi.strip().split(' ')
        else:
            e1_multi_tokens = self.tokenizer(self.e1_multi)

        if self.tokenizer is None:
            rel_tokens = self.rel.strip().split(self.split_token)
        else:
            rel_tokens = self.tokenizer(self.rel)

        if self.tokenizer is None:
            rel_eval_tokens = self.rel_eval.strip().split(self.split_token)
        else:
            rel_eval_tokens = self.tokenizer(self.rel_eval)

        if self.share_vocab:
            return e1_tokens + e2_tokens + e1_multi_tokens + e2_multi_tokens + rel_tokens + rel_eval_tokens
        else:
            return e1_tokens + e2_tokens + e1_multi_tokens + e2_multi_tokens, rel_tokens + rel_eval_tokens


class KGCompletionDataset(Dataset):
    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, share_vocab=True,
                 edge_strategy=None, **kwargs):
        self.data_item_type = KGDataItem
        self.share_vocab = share_vocab  # share vocab between entity and relation
        self.edge_strategy = edge_strategy
        super(KGCompletionDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.

        For Text2TextDataset, the format of the input file should contain lines of input, each line representing one
        record of data. The input and output is separated by a tab(\t).

        Examples
        --------
        {"e1": "person84", "e2": "person85", "rel": "term21", "rel_eval": "term21_reverse",
        "e2_multi1": "person85",
        "e2_multi2": "person74 person84 person55 person96 person66 person57"}

        {"e1": "person20", "e2": "person90", "rel": "term11", "rel_eval": "term11_reverse",
        "e2_multi1": "person29 person82 person85 person77 person73 person63 person34 person86 person4
        person83 person46 person16 person48 person17 person59 person80 person50 person90",
        "e2_multi2": "person29 person82 person2 person20 person83 person46 person80"}

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
                line_dict = json.loads(line)
                data_item = KGDataItem(e1=line_dict['e1'],
                                                      e2=line_dict['e2'],
                                                      rel=line_dict['rel'],
                                                      rel_eval=line_dict['rel_eval'],
                                                      e2_multi=line_dict['e2_multi1'],
                                                      e1_multi=line_dict['e2_multi2'],
                                                      share_vocab=self.share_vocab,
                                                      split_token=self.split_token)
                data.append(data_item)

                if 'train' in file_path:
                    for e2 in line_dict['e2_multi1'].split(' '):
                        triple = [line_dict['e1'], line_dict['rel'], e2]
                        self.build_parsed_results(triple)

                    for e1 in line_dict['e2_multi2'].split(' '):
                        triple = [line_dict['e2'], line_dict['rel_eval'], e1]
                        self.build_parsed_results(triple)

        return data

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        vocab_model = VocabModel.build(
            saved_vocab_file=os.path.join(self.processed_dir, self.processed_file_names['vocab']),
            data_set=data_for_vocab,
            tokenizer=self.tokenizer,
            lower_case=self.lower_case,
            max_word_vocab_size=None,
            min_word_vocab_freq=1,
            pretrained_word_emb_file=self.pretrained_word_emb_file,
            word_emb_size=300)
        self.vocab_model = vocab_model

        return self.vocab_model

    def build_parsed_results(self, triple):
        if self.edge_strategy is None:
            if triple[0] not in self.graph_nodes:
                self.graph_nodes.append(triple[0])

            if triple[2] not in self.graph_nodes:
                self.graph_nodes.append(triple[2])

            if triple[1] not in self.graph_edges:
                self.graph_edges.append(triple[1])

            triple_info = {'edge_tokens': triple[1],
                           'src': {
                               'tokens': triple[0],
                               'id': self.graph_nodes.index(triple[0])
                           },
                           'tgt': {
                               'tokens': triple[2],
                               'id': self.graph_nodes.index(triple[2])
                           }}
            if triple_info not in self.parsed_results['graph_content']:
                self.parsed_results['graph_content'].append(triple_info)
        elif self.edge_strategy == "as_node":
            if triple[0] not in self.graph_nodes:
                self.graph_nodes.append(triple[0])

            if triple[1] not in self.graph_nodes:
                self.graph_nodes.append(triple[1])

            if triple[2] not in self.graph_nodes:
                self.graph_nodes.append(triple[2])

            triple_info_0_1 = {'edge_tokens': [],
                               'src': {
                                   'tokens': triple[0],
                                   'id': self.graph_nodes.index(triple[0]),
                                   'type': 'ent_node'
                               },
                               'tgt': {
                                   'tokens': triple[1],
                                   'id': self.graph_nodes.index(triple[1]),
                                   'type': 'edge_node'
                               }}

            triple_info_1_2 = {'edge_tokens': [],
                               'src': {
                                   'tokens': triple[1],
                                   'id': self.graph_nodes.index(triple[1]),
                                   'type': 'edge_node'
                               },
                               'tgt': {
                                   'tokens': triple[2],
                                   'id': self.graph_nodes.index(triple[2]),
                                   'type': 'ent_node'
                               }}

            if triple_info_0_1 not in self.parsed_results['graph_content']:
                self.parsed_results['graph_content'].append(triple_info_0_1)
            if triple_info_1_2 not in self.parsed_results['graph_content']:
                self.parsed_results['graph_content'].append(triple_info_1_2)

        return

    def build_topology(self):
        # self.data: [KGDataItem] = self.build_dataitem(self.raw_file_paths)
        self.KG_graph = GraphData()
        self.parsed_results['node_num'] = len(self.graph_nodes)
        self.parsed_results['graph_nodes'] = self.graph_nodes
        self.KG_graph = IEBasedGraphConstruction._construct_static_graph(self.parsed_results,
                                                                         edge_strategy=self.edge_strategy)
        self.KG_graph.graph_attributes['num_entities'] = len(self.graph_nodes)
        self.KG_graph.graph_attributes['num_relations'] = len(self.graph_edges)
        self.KG_graph.graph_attributes['graph_nodes'] = self.graph_nodes
        self.KG_graph.graph_attributes['graph_edges'] = self.graph_edges

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths.values()]):
            if 'val_split_ratio' in self.__dict__:
                UserWarning(
                    "Loading existing processed files on disk. Your `val_split_ratio` might not work since the data have"
                    "already been split.")
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        self.graph_nodes = []
        self.graph_edges = []
        """
        `self.parsed_results` is an intermediate dict that contains all the information of the KG graph.
        `self.parsed_results['graph_content']` is a list of dict.

        Each dict in `self.parsed_results['graph_content']` contains information about a triple
        (src_ent, rel, tgt_ent).

        `self.parsed_results['graph_nodes']` contains all nodes in the KG graph.
        `self.parsed_results['node_num']` is the number of nodes in the KG graph.
        """
        self.parsed_results = {}
        self.parsed_results['graph_content'] = []

        self.read_raw_data()
        self.build_topology()
        self.build_vocab()

        self.vec_graph = False
        self.vectorization(self.train)
        self.vectorization(self.test)
        if 'val' in self.__dict__:
            self.vectorization(self.val)

        data_to_save = {'train': self.train, 'test': self.test}
        if 'val' in self.__dict__:
            data_to_save['val'] = self.val
        torch.save(data_to_save, self.processed_file_paths['data'])

    def vectorization(self, data_items):
        if self.vec_graph==False:
            graph: GraphData = self.KG_graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features['token_id'] = token_matrix
            self.vec_graph = True
            torch.save(self.KG_graph, os.path.join(self.processed_dir, self.processed_file_names['KG_graph']))

        for item in data_items:
            e1 = item.e1
            item.e1_tensor = torch.tensor(self.graph_nodes.index(e1), dtype=torch.long)

            e2 = item.e2
            item.e2_tensor = torch.tensor(self.graph_nodes.index(e2), dtype=torch.long)

            rel = item.rel
            if self.edge_strategy == "as_node":
                item.rel_tensor = torch.tensor(self.graph_nodes.index(rel), dtype=torch.long)
            else:
                item.rel_tensor = torch.tensor(self.graph_edges.index(rel), dtype=torch.long)

            rel_eval = item.rel_eval
            item.rel_eval_tensor = torch.tensor(self.graph_edges.index(rel_eval), dtype=torch.long)


            e2_multi = item.e2_multi
            item.e2_multi_tensor = torch.zeros(1, len(self.graph_nodes)).\
                scatter_(1,
                         torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long).view(1,
                                                                                                                    -1),
                         torch.ones(1, len(e2_multi.split()))).squeeze()

            item.e2_multi_tensor_idx = torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long)

            e1_multi = item.e1_multi
            item.e1_multi_tensor = torch.zeros(1, len(self.graph_nodes)). \
                scatter_(1,
                         torch.tensor([self.graph_nodes.index(i) for i in e1_multi.split()], dtype=torch.long).view(1,
                                                                                                                    -1),
                         torch.ones(1, len(e1_multi.split()))).squeeze()
            item.e1_multi_tensor_idx = torch.tensor([self.graph_nodes.index(i) for i in e1_multi.split()],
                                                            dtype=torch.long)



    # @staticmethod
    # def collate_fn(data_list: [KGDataItem]):
    #     # graph_data = [item.graph for item in data_list]
    #     e1 = torch.tensor([item.e1_tensor for item in data_list])
    #     rel = torch.tensor([item.rel_tensor for item in data_list])
    #     # e2_multi_tensor_idx = torch.tensor([item.e2_multi_tensor_idx for item in data_list])
    #
    #     e2_multi_tensor_idx_len = [item.e2_multi_tensor_idx.shape[0] for item in data_list]
    #     max_e2_multi_tensor_idx_len = max(e2_multi_tensor_idx_len)
    #     e2_multi_tensor_idx_pad = []
    #     for item in data_list:
    #         if item.e2_multi_tensor_idx.shape[0] < max_e2_multi_tensor_idx_len:
    #             need_pad_length = max_e2_multi_tensor_idx_len - item.e2_multi_tensor_idx.shape[0]
    #             pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
    #             e2_multi_tensor_idx_pad.append(torch.cat((item.e2_multi_tensor_idx, pad.long()), dim=0).unsqueeze(0))
    #         elif item.e2_multi_tensor_idx.shape[0] == max_e2_multi_tensor_idx_len:
    #             e2_multi_tensor_idx_pad.append(item.e2_multi_tensor_idx.unsqueeze(0))
    #         else:
    #             raise RuntimeError("Size mismatch error")
    #
    #     e2_multi_tensor_idx = torch.cat(e2_multi_tensor_idx_pad, dim=0)
    #
    #     # do padding here
    #     e2_multi_len = [item.e2_multi_tensor.shape[0] for item in data_list]
    #     max_e2_multi_len = max(e2_multi_len)
    #     e2_multi_pad = []
    #     for item in data_list:
    #         if item.e2_multi_tensor.shape[0] < max_e2_multi_len:
    #             need_pad_length = max_e2_multi_len - item.e2_multi_tensor.shape[0]
    #             pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
    #             e2_multi_pad.append(torch.cat((item.e2_multi_tensor, pad.long()), dim=0).unsqueeze(0))
    #         elif item.e2_multi_tensor.shape[0] == max_e2_multi_len:
    #             e2_multi_pad.append(item.e2_multi_tensor.unsqueeze(0))
    #         else:
    #             raise RuntimeError("Size mismatch error")
    #
    #     e2_multi = torch.cat(e2_multi_pad, dim=0)
    #
    #     # # do padding here
    #     # seq_len = [item.output_tensor.shape[0] for item in data_list]
    #     # max_seq_len = max(seq_len)
    #     # tgt_seq_pad = []
    #     # for item in data_list:
    #     #     if item.output_tensor.shape[0] < max_seq_len:
    #     #         need_pad_length = max_seq_len - item.output_tensor.shape[0]
    #     #         pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
    #     #         tgt_seq_pad.append(torch.cat((item.output_tensor, pad.long()), dim=0).unsqueeze(0))
    #     #     elif item.output_tensor.shape[0] == max_seq_len:
    #     #         tgt_seq_pad.append(item.output_tensor.unsqueeze(0))
    #     #     else:
    #     #         raise RuntimeError("Size mismatch error")
    #     #
    #     # tgt_seq = torch.cat(tgt_seq_pad, dim=0)
    #     return [e1, rel, e2_multi, e2_multi_tensor_idx]

    @staticmethod
    def collate_fn(data_list: [KGDataItem]):
        # graph_data = [item.graph for item in data_list]
        e1 = torch.tensor([item.e1_tensor for item in data_list])
        e2 = torch.tensor([item.e2_tensor for item in data_list])
        rel = torch.tensor([item.rel_tensor for item in data_list])
        rel_eval = torch.tensor([item.rel_eval_tensor for item in data_list])

        # do padding here
        e1_multi_tensor_idx_len = [item.e1_multi_tensor_idx.shape[0] for item in data_list]
        max_e1_multi_tensor_idx_len = max(e1_multi_tensor_idx_len)
        e1_multi_tensor_idx_pad = []
        for item in data_list:
            if item.e1_multi_tensor_idx.shape[0] < max_e1_multi_tensor_idx_len:
                need_pad_length = max_e1_multi_tensor_idx_len - item.e1_multi_tensor_idx.shape[0]
                pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
                e1_multi_tensor_idx_pad.append(torch.cat((item.e1_multi_tensor_idx, pad.long()), dim=0).unsqueeze(0))
            elif item.e1_multi_tensor_idx.shape[0] == max_e1_multi_tensor_idx_len:
                e1_multi_tensor_idx_pad.append(item.e1_multi_tensor_idx.unsqueeze(0))
            else:
                raise RuntimeError("Size mismatch error")

        e1_multi_tensor_idx = torch.cat(e1_multi_tensor_idx_pad, dim=0)

        e1_multi_len = [item.e1_multi_tensor.shape[0] for item in data_list]
        max_e1_multi_len = max(e1_multi_len)
        e1_multi_pad = []
        for item in data_list:
            if item.e1_multi_tensor.shape[0] < max_e1_multi_len:
                need_pad_length = max_e1_multi_len - item.e1_multi_tensor.shape[0]
                pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
                e1_multi_pad.append(torch.cat((item.e1_multi_tensor, pad.long()), dim=0).unsqueeze(0))
            elif item.e1_multi_tensor.shape[0] == max_e1_multi_len:
                e1_multi_pad.append(item.e1_multi_tensor.unsqueeze(0))
            else:
                raise RuntimeError("Size mismatch error")

        e1_multi = torch.cat(e1_multi_pad, dim=0)

        # do padding here
        e2_multi_tensor_idx_len = [item.e2_multi_tensor_idx.shape[0] for item in data_list]
        max_e2_multi_tensor_idx_len = max(e2_multi_tensor_idx_len)
        e2_multi_tensor_idx_pad = []
        for item in data_list:
            if item.e2_multi_tensor_idx.shape[0] < max_e2_multi_tensor_idx_len:
                need_pad_length = max_e2_multi_tensor_idx_len - item.e2_multi_tensor_idx.shape[0]
                pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
                e2_multi_tensor_idx_pad.append(torch.cat((item.e2_multi_tensor_idx, pad.long()), dim=0).unsqueeze(0))
            elif item.e2_multi_tensor_idx.shape[0] == max_e2_multi_tensor_idx_len:
                e2_multi_tensor_idx_pad.append(item.e2_multi_tensor_idx.unsqueeze(0))
            else:
                raise RuntimeError("Size mismatch error")

        e2_multi_tensor_idx = torch.cat(e2_multi_tensor_idx_pad, dim=0)

        e2_multi_len = [item.e2_multi_tensor.shape[0] for item in data_list]
        max_e2_multi_len = max(e2_multi_len)
        e2_multi_pad = []
        for item in data_list:
            if item.e2_multi_tensor.shape[0] < max_e2_multi_len:
                need_pad_length = max_e2_multi_len - item.e2_multi_tensor.shape[0]
                pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
                e2_multi_pad.append(torch.cat((item.e2_multi_tensor, pad.long()), dim=0).unsqueeze(0))
            elif item.e2_multi_tensor.shape[0] == max_e2_multi_len:
                e2_multi_pad.append(item.e2_multi_tensor.unsqueeze(0))
            else:
                raise RuntimeError("Size mismatch error")

        e2_multi = torch.cat(e2_multi_pad, dim=0)

        return [e1, rel, e2_multi, e2_multi_tensor_idx, e2, rel_eval, e1_multi, e1_multi_tensor_idx]


class Text2LabelDataset(Dataset):
    def __init__(self, root_dir, topology_builder, topology_subdir, **kwargs):
        self.data_item_type = Text2LabelDataItem
        super(Text2LabelDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.

        For Text2LabelDataset, the format of the input file should contain lines of input, each line representing one
        record of data. The input and output is separated by a tab(\t).

        Examples
        --------
        input: How far is it from Denver to Aspen ?    NUM

        DataItem: input_text="How far is it from Denver to Aspen ?", output_label="NUM"

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
                data_item = Text2LabelDataItem(input_text=input.strip(), output_label=output.strip(), tokenizer=self.tokenizer)
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
                                       max_word_vocab_size=self.max_word_vocab_size,
                                       min_word_vocab_freq=self.min_word_vocab_freq,
                                       pretrained_word_emb_file=self.pretrained_word_emb_file,
                                       word_emb_size=self.word_emb_size,
                                       share_vocab=True)
        self.vocab_model = vocab_model

        # label encoding
        all_labels = {item.output_label for item in self.train + self.test}
        if 'val' in self.__dict__:
            all_labels = all_labels.union({item.output_label for item in self.val})

        self.le = preprocessing.LabelEncoder()
        self.le.fit(list(all_labels))
        self.num_classes = len(self.le.classes_)

    def vectorization(self, data_items):
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.LongTensor(token_matrix)
            graph.node_features['token_id'] = token_matrix

            item.output = self.le.transform([item.output_label])[0]

    @staticmethod
    def collate_fn(data_list: [Text2LabelDataItem]):
        graph_data = [item.graph for item in data_list]

        tgt = [item.output for item in data_list]
        tgt_tensor = torch.LongTensor(tgt)

        return [graph_data, tgt_tensor]
