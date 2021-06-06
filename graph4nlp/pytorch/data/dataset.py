import abc
import json
import os
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np
import stanfordcorenlp
import torch.utils.data
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from copy import deepcopy

from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
from ..data.data import GraphData, to_batch
from ..modules.graph_construction.base import GraphConstructionBase
from ..modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from ..modules.utils.tree_utils import Tree
from ..modules.utils.tree_utils import Vocab as VocabForTree
from ..modules.utils.vocab_utils import VocabModel, Vocab


class DataItem(object):
    def __init__(self, input_text, tokenizer):
        self.input_text = input_text
        self.tokenizer = tokenizer
        pass

    @abc.abstractmethod
    def extract(self):
        raise NotImplementedError


class Text2TextDataItem_seq2seq(DataItem):
    def __init__(self, input_text, output_text, tokenizer, share_vocab=True):
        super(Text2TextDataItem_seq2seq, self).__init__(input_text, tokenizer)
        self.output_text = output_text
        self.share_vocab = share_vocab

    def extract(self, lower_case=True):
        """
        Returns
        -------
        Input tokens and output tokens
        """

        if lower_case:
            self.input_text = self.input_text.lower()

        if self.tokenizer is None:
            input_tokens = self.input_text.strip().split(' ')
        else:
            input_tokens = self.tokenizer(self.input_text)


        if lower_case:
            self.output_text = self.output_text.lower()

        if self.tokenizer is None:
            output_tokens = self.output_text.strip().split(' ')
        else:
            if '<t>' in self.output_text:
                output_text = self.output_text.replace('<t>','').replace('</t>','')
            output_tokens = self.tokenizer(output_text)
            output_tokens = output_tokens + ['<t>','</t>']

        if self.share_vocab:
            return input_tokens + output_tokens
        else:
            return input_tokens, output_tokens


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
            tokenized_token = self.tokenizer(g.node_attributes[i]['token'])
            input_tokens.extend(tokenized_token)

        output_tokens = self.tokenizer(self.output_text)

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


class DoubleText2TextDataItem(DataItem):
    def __init__(self, input_text, input_text2, output_text, tokenizer, share_vocab=True):
        super(DoubleText2TextDataItem, self).__init__(input_text, tokenizer)
        self.input_text2 = input_text2
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
                tokenized_token = g.node_attributes[i]['token'].strip().split()
            else:
                tokenized_token = self.tokenizer(g.node_attributes[i]['token'])

            input_tokens.extend(tokenized_token)

        if self.tokenizer is None:
            input_tokens.extend(self.input_text2.strip().split())
            output_tokens = self.output_text.strip().split()
        else:
            input_tokens.extend(self.tokenizer(self.input_text2))
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
                 pretrained_word_emb_name="840B",
                 pretrained_word_emb_url=None,
                 target_pretrained_word_emb_name=None,
                 target_pretrained_word_emb_url=None,
                 pretrained_word_emb_cache_dir=".vector_cache/",
                 max_word_vocab_size=None,
                 min_word_vocab_freq=1,
                 use_val_for_vocab=False,
                 seed=1234,
                 thread_number=4,
                 port=9000,
                 timeout=15000,
                 **kwargs):
        """

        Parameters
        ----------
        root: str
            The path of the data root.
        topology_builder: GraphConstructionBase
            The initial graph topology builder.
        topology_subdir: str
            The name of the data folder.
        tokenizer: function, default=nltk.word_tokenize
            The word tokenizer.
        lower_case: bool, default=True
            Whether to use lower-case option.
        pretrained_word_emb_name: str, optional, default="840B"
            The name of pretrained word embedding in ``torchtext``.
            If it is set ``None``, we will randomly set the initial word embedding values.
        pretrained_word_emb_url: str optional, default: ``None``
            The url for downloading pretrained word embedding.
            Note that we only prepare the default ``url`` for English with ``pretrained_word_emb_name`` as ``"42B"``, ``"840B"``, 'twitter.27B' and '6B'.
        target_pretrained_word_emb_name: str, optional, default=None
            The name of pretrained word embedding in ``torchtext`` for target language.
            If it is set ``None``, we will use ``pretrained_word_emb_name``.
        target_pretrained_word_emb_url: str optional, default: ``None``
            The url for downloading pretrained word embedding for target language.
            Note that we only prepare the default ``url`` for English with ``pretrained_word_emb_name`` as ``"42B"``, ``"840B"``, 'twitter.27B' and '6B'.
        pretrained_word_emb_cache_dir: str, optional, default: ``".vector_cache/"``
            The path of directory saving the temporary word embedding file.
        use_val_for_vocab: bool, default=False
            Whether to add val split in the final split.
        seed: int, default=1234
            The seed for random function.
        thread_number: int, default=4
            The thread number for building initial graph. For most case, it may be the number of your CPU cores.
        port: int, default=9000
            The port for stanfordcorenlp.
        timeout: int, default=15000
            The timeout for stanfordcorenlp.
        kwargs
        """
        super(Dataset, self).__init__()

        self.root = root  # The root directory where the dataset is located.
        self.seed = seed

        # stanfordcorenlp hyper-parameter
        self.thread_number = thread_number
        self.port = port
        self.timeout = timeout

        # Processing-specific attributes
        self.tokenizer = tokenizer
        self.lower_case = lower_case
        self.pretrained_word_emb_name = pretrained_word_emb_name
        self.pretrained_word_emb_url = pretrained_word_emb_url
        self.target_pretrained_word_emb_name = target_pretrained_word_emb_name
        self.target_pretrained_word_emb_url = target_pretrained_word_emb_url
        self.pretrained_word_emb_cache_dir= pretrained_word_emb_cache_dir
        self.max_word_vocab_size = max_word_vocab_size
        self.min_word_vocab_freq = min_word_vocab_freq

        # self.pretrained_word_emb_file = pretrained_word_emb_file
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
                random.seed(self.seed)
                old_train_set = self.train
                random.shuffle(old_train_set)
                self.val = old_train_set[new_train_length:]
                self.train = old_train_set[:new_train_length]

    @staticmethod
    def _build_topology_process(data_items, topology_builder,
                                graph_type, dynamic_graph_type, dynamic_init_topology_builder,
                                merge_strategy, edge_strategy, dynamic_init_topology_aux_args,
                                lower_case, tokenizer, port, timeout):

        ret = []
        if graph_type == 'static':
            print('Connecting to stanfordcorenlp server...')
            processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=port, timeout=timeout)

            if topology_builder == IEBasedGraphConstruction:
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
            elif topology_builder == DependencyBasedGraphConstruction:
                processor_args = {
                    'annotators': 'ssplit,tokenize,depparse',
                    "tokenize.options":
                        "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                    "tokenize.whitespace": True,
                    'ssplit.isOneSentence': True,
                    'outputFormat': 'json'
                }
            elif topology_builder == ConstituencyBasedGraphConstruction:
                processor_args = {
                    'annotators': "tokenize,ssplit,pos,parse",
                    "tokenize.options":
                        "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                    "tokenize.whitespace": True,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
            else:
                raise NotImplementedError
            print('CoreNLP server connected.')
            pop_idxs = []
            for cnt, item in enumerate(data_items):
                if cnt % 1000 == 0:
                    print("Port {}, processing: {} / {}".format(port, cnt, len(data_items)))
                # graph = topology_builder.topology(raw_text_data=item.input_text,
                #                                     nlp_processor=processor,
                #                                     processor_args=processor_args,
                #                                     merge_strategy=merge_strategy,
                #                                     edge_strategy=edge_strategy,
                #                                     verbase=False)

                try:
                    graph = topology_builder.topology(raw_text_data=item.input_text,
                                                      nlp_processor=processor,
                                                      processor_args=processor_args,
                                                      merge_strategy=merge_strategy,
                                                      edge_strategy=edge_strategy,
                                                      verbase=False)
                    item.graph = graph
                except Exception as msg:
                    pop_idxs.append(cnt)
                    item.graph = None
                    # import traceback
                    # traceback.print_exc()
                    # print(item.input_text)
                    # exit(0)
                    # print(msg)
                    warnings.warn(RuntimeWarning(msg))
                ret.append(item)
            ret = [x for idx, x in enumerate(ret) if idx not in pop_idxs]
        elif graph_type == 'dynamic':
            if dynamic_graph_type == 'node_emb':
                for item in data_items:
                    graph = topology_builder.init_topology(item.input_text,
                                                           lower_case=lower_case,
                                                           tokenizer=tokenizer)
                    item.graph = graph
                    ret.append(item)
            elif dynamic_graph_type == 'node_emb_refined':
                if dynamic_init_topology_builder in (
                IEBasedGraphConstruction, DependencyBasedGraphConstruction, ConstituencyBasedGraphConstruction):
                    print('Connecting to stanfordcorenlp server...')
                    processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=port, timeout=timeout)

                    if dynamic_init_topology_builder == IEBasedGraphConstruction:
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
                    elif dynamic_init_topology_builder == DependencyBasedGraphConstruction:
                        processor_args = {
                            'annotators': 'ssplit,tokenize,depparse',
                            "tokenize.options":
                                "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                    elif dynamic_init_topology_builder == ConstituencyBasedGraphConstruction:
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
                pop_idxs = []
                for idx, item in enumerate(data_items):
                    try:
                        graph = topology_builder.init_topology(item.input_text,
                                                               dynamic_init_topology_builder=dynamic_init_topology_builder,
                                                               lower_case=lower_case,
                                                               tokenizer=tokenizer,
                                                               nlp_processor=processor,
                                                               processor_args=processor_args,
                                                               merge_strategy=merge_strategy,
                                                               edge_strategy=edge_strategy,
                                                               verbase=False,
                                                               dynamic_init_topology_aux_args=dynamic_init_topology_aux_args)

                        item.graph = graph
                    except Exception as msg:
                        pop_idxs.append(idx)
                        item.graph = None
                        warnings.warn(RuntimeWarning(msg))
                    ret.append(item)
                ret = [x for idx, x in enumerate(ret) if idx not in pop_idxs]
            else:
                raise RuntimeError('Unknown dynamic_graph_type: {}'.format(dynamic_graph_type))

        else:
            raise NotImplementedError('Currently only static and dynamic are supported!')

        return ret

    def build_topology(self, data_items):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        total = len(data_items)
        thread_number = min(total, self.thread_number)
        pool = Pool(thread_number)
        res_l = []
        for i in range(thread_number):
            start_index = total * i // thread_number
            end_index = total * (i + 1) // thread_number

            """
            data_items, topology_builder,
                                graph_type, dynamic_graph_type, dynamic_init_topology_builder,
                                merge_strategy, edge_strategy, dynamic_init_topology_aux_args,
                                lower_case, tokenizer, port, timeout
            """
            r = pool.apply_async(self._build_topology_process,
                                 args=(data_items[start_index:end_index], self.topology_builder, self.graph_type,
                                       self.dynamic_graph_type, self.dynamic_init_topology_builder,
                                       self.merge_strategy, self.edge_strategy, self.dynamic_init_topology_aux_args,
                                       self.lower_case, self.tokenizer, self.port, self.timeout))
            res_l.append(r)
        pool.close()
        pool.join()

        data_items = []
        for i in range(thread_number):
            res = res_l[i].get()
            for data in res:
                if data.graph is not None:
                    data_items.append(data)

        return data_items

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
                                       min_word_vocab_freq=self.min_word_vocab_freq,
                                       share_vocab=self.share_vocab,
                                       pretrained_word_emb_name=self.pretrained_word_emb_name,
                                       pretrained_word_emb_url=self.pretrained_word_emb_url,
                                       pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
                                       target_pretrained_word_emb_name=self.target_pretrained_word_emb_name,
                                       target_pretrained_word_emb_url=self.target_pretrained_word_emb_url,
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

        self.train = self.build_topology(self.train)

        self.test = self.build_topology(self.test)
        if 'val' in self.__dict__:
            self.val = self.build_topology(self.val)

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
            tgt_token_id = self.vocab_model.out_word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(self.vocab_model.out_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            item.output_np = tgt_token_id

    @staticmethod
    def collate_fn(data_list: [Text2TextDataItem]):
        graph_list = [item.graph for item in data_list]
        graph_data = to_batch(graph_list)

        output_numpy = [deepcopy(item.output_np) for item in data_list]
        output_str = [deepcopy(item.output_text.lower().strip()) for item in data_list]
        output_pad = pad_2d_vals_no_size(output_numpy)

        tgt_seq = torch.from_numpy(output_pad).long()
        return {
            "graph_data": graph_data,
            "tgt_seq": tgt_seq,
            "output_str": output_str
        }


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
                data_item = Text2TreeDataItem(input_text=input, output_text=output, output_tree=None,
                                              tokenizer=self.tokenizer,
                                              share_vocab=self.share_vocab)
                data.append(data_item)
        return data

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        src_vocab_model = VocabForTree(lower_case=self.lower_case,
                                       pretrained_word_emb_name=self.pretrained_word_emb_name,
                                       pretrained_word_emb_url=self.pretrained_word_emb_url,
                                       pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
                                       embedding_dims=self.enc_emb_size)
        tgt_vocab_model = VocabForTree(lower_case=self.lower_case,
                                       pretrained_word_emb_name=self.pretrained_word_emb_name,
                                       pretrained_word_emb_url=self.pretrained_word_emb_url,
                                       pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
                                       embedding_dims=self.dec_emb_size)

        if self.share_vocab:
            all_words = Counter()
        else:
            all_words = [Counter(), Counter()]

        for instance in data_for_vocab:
            extracted_tokens = instance.extract()
            if self.share_vocab:
                all_words.update(extracted_tokens[0])
                all_words.update(extracted_tokens[1])
            else:
                all_words[0].update(extracted_tokens[0])
                all_words[1].update(extracted_tokens[1])

        if self.share_vocab:
            src_vocab_model.init_from_list(list(all_words.items()), min_freq=self.min_word_vocab_freq, max_vocab_size=self.max_word_vocab_size)
            tgt_vocab_model = src_vocab_model
        else:
            src_vocab_model.init_from_list(list(all_words[0].items()), min_freq=self.min_word_vocab_freq, max_vocab_size=self.max_word_vocab_size)
            tgt_vocab_model.init_from_list(list(all_words[1].items()), min_freq=self.min_word_vocab_freq, max_vocab_size=self.max_word_vocab_size)

        self.src_vocab_model = src_vocab_model
        self.tgt_vocab_model = tgt_vocab_model
        if self.share_vocab:
            self.share_vocab_model = src_vocab_model

        return self.src_vocab_model

    def vectorization(self, data_items):
        """For tree decoder we also need the vectorize the tree output."""
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
        # remove the deepcopy
        graph_data = [item.graph for item in data_list]
        graph_data = to_batch(graph_data)

        output_tree_list = [deepcopy(item.output_tree) for item in data_list]
        original_sentence_list = [deepcopy(item.output_text) for item in data_list]

        return {
            "graph_data": graph_data,
            "dec_tree_batch": output_tree_list,
            "original_dec_tree_batch": original_sentence_list
        }


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
            for line in f:
                input, output = line.split('\t')
                data_item = Text2LabelDataItem(input_text=input.strip(), output_label=output.strip(),
                                               tokenizer=self.tokenizer)
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
                                       pretrained_word_emb_name=self.pretrained_word_emb_name,
                                       pretrained_word_emb_url=self.pretrained_word_emb_url,
                                       pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
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
        graph_list = [item.graph for item in data_list]
        graph_data = to_batch(graph_list)

        tgt = [deepcopy(item.output) for item in data_list]
        tgt_tensor = torch.LongTensor(tgt)

        return {
            "graph_data": graph_data,
            "tgt_tensor": tgt_tensor
        }


class DoubleText2TextDataset(Dataset):
    def __init__(self, root_dir, topology_builder, topology_subdir, share_vocab=True, **kwargs):
        self.data_item_type = DoubleText2TextDataItem
        self.share_vocab = share_vocab
        super(DoubleText2TextDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.

        For DoubleText2TextDataset, the format of the input file should contain lines of input, each line representing one
        record of data. The input and output is separated by a tab(\t).
        # TODO: update example

        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )

        DataItem: input_text="list job use languageid0", input_text2="list job use languageid0", output_text="job ( ANS ) , language ( ANS , languageid0 )"

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
            for line in f:
                if self.lower_case:
                    line = line.lower().strip()
                input, input2, output = line.split('\t')
                data_item = DoubleText2TextDataItem(input_text=input.strip(), input_text2=input2.strip(),
                                                    output_text=output.strip(), tokenizer=self.tokenizer,
                                                    share_vocab=self.share_vocab)
                data.append(data_item)
        return data

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

            item.input_text2 = self.tokenizer(item.input_text2)
            input_token_id2 = self.vocab_model.in_word_vocab.to_index_sequence_for_list(item.input_text2)
            input_token_id2 = np.array(input_token_id2)
            item.input_np2 = input_token_id2

            if self.lower_case:
                item.output_text = item.output_text.lower()

            item.output_text = self.tokenizer(item.output_text)

            tgt_token_id = self.vocab_model.in_word_vocab.to_index_sequence_for_list(item.output_text)
            tgt_token_id.append(self.vocab_model.in_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            item.output_np = tgt_token_id
            item.output_text = ' '.join(item.output_text)

    @staticmethod
    def collate_fn(data_list: [Text2TextDataItem]):
        graph_data = []
        input_tensor2, input_length2, input_text2 = [], [], []
        tgt_tensor, tgt_text = [], []
        for item in data_list:
            graph_data.append(item.graph)
            input_tensor2.append(deepcopy(item.input_np2))
            input_length2.append(len(item.input_np2))
            input_text2.append(deepcopy(item.input_text2))
            tgt_tensor.append(deepcopy(item.output_np))
            tgt_text.append(deepcopy(item.output_text))

        input_tensor2 = torch.LongTensor(pad_2d_vals_no_size(input_tensor2))
        input_length2 = torch.LongTensor(input_length2)
        tgt_tensor = torch.LongTensor(pad_2d_vals_no_size(tgt_tensor))
        graph_data = to_batch(graph_data)

        return {'graph_data': graph_data,
                'input_tensor2': input_tensor2,
                'input_text2': input_text2,
                'tgt_tensor': tgt_tensor,
                'tgt_text': tgt_text,
                'input_length2': input_length2}


class SequenceLabelingDataset(Dataset):
    def __init__(self, root_dir, topology_builder, topology_subdir, tag_types, **kwargs):
        self.data_item_type = SequenceLabelingDataItem
        self.tag_types = tag_types
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
        data = []
        input = []
        output = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 1 and line[0] != '-':
                    if line[0] != '.':
                        input.append(line.strip().split(' ')[0])
                        output.append(line.strip().split(' ')[-1])
                    if line[0] == '.':
                        input.append(line.strip().split(' ')[0])
                        output.append(line.strip().split(' ')[-1])
                        if len(input) >= 2:
                            data_item = SequenceLabelingDataItem(input_text=input, output_tags=output,
                                                                 tokenizer=self.tokenizer)
                            data.append(data_item)
                            input = []
                            output = []

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
                                       pretrained_word_emb_name=self.pretrained_word_emb_name,
                                       pretrained_word_emb_url=self.pretrained_word_emb_url,
                                       pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
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
            tgt_tag_id = [self.tag_types.index(tgt_.strip()) for tgt_ in tgt]

            tgt_tag_id = torch.tensor(tgt_tag_id)
            item.output_id = tgt_tag_id

    @staticmethod
    def collate_fn(data_list: [SequenceLabelingDataItem]):
        
        graph_list= [item.graph for item in data_list]
        graph_data=to_batch(graph_list)
        tgt_tag = [deepcopy(item.output_id) for item in data_list]

        return {"graph_data": graph_data,
            "tgt_tag": tgt_tag}
