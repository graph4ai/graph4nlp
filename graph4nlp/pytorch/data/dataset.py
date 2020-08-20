import abc
import copy
import os

import numpy as np
import random
import stanfordcorenlp
import torch.utils.data
from nltk.tokenize import word_tokenize

from ..data.data import GraphData
from ..modules.utils.vocab_utils import VocabModel, Vocab


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
    def vectorization(self):
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

        self.data = {}  # The dictionary maintaining data and indices.
        self.split_ids = {}  # The dictionary indicating the indices of each split subset.
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
        for split_name, file_path in self.raw_file_paths.items():
            self.split_ids[split_name] = self.parse_file(file_path)
        torch.save(self.split_ids, self.processed_file_paths['split_ids'])

    def build_topology(self):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        if self.graph_type == 'static':
            processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)
            for item in self.data.values():
                graph = self.topology_builder.topology(raw_text_data=item.input_text, nlp_processor=processor,
                                                       merge_strategy=self.merge_strategy,
                                                       edge_strategy=self.edge_strategy)
                item.graph = graph
        elif self.graph_type == 'dynamic':
            # TODO: Implement this
            pass
        else:
            raise NotImplementedError('Currently only static and dynamic are supported!')

    def build_vocab(self):
        """
        Build the vocabulary. If `self.use_val_for_vocab` is `True`, use both training set and validation set for building
        the vocabulary. Otherwise only the training set is used.

        """
        data_for_vocab = [self.data[idx] for idx in self.split_ids['train']]
        if self.use_val_for_vocab:
            data_for_vocab += [self.data[idx] for idx in self.split_ids['val']]

        vocab_model = VocabModel.build(saved_vocab_file=self.processed_file_paths['vocab'],
                                       data_set=data_for_vocab,
                                       tokenizer=self.tokenizer,
                                       lower_case=self.lower_case,
                                       max_word_vocab_size=None,
                                       min_word_vocab_freq=1,
                                       pretrained_word_emb_file=self.pretrained_word_emb_file,
                                       word_emb_size=300)
        self.vocab_model = vocab_model

        return self.vocab_model

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths.values()]):
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        self.read_raw_data()
        self.build_topology()
        self.build_vocab()
        self.vectorization()

    @property
    def indices(self):
        if self.__indices__ is not None:
            return self.__indices__
        else:
            self.__dict__['indices'] = list(self.data.keys())
            return list(self.data.keys())

    def index_select(self, idx):
        indices = self.indices

        if isinstance(idx, slice):
            indices = indices[idx]
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                return self.index_select(idx.tolist())
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.index_select(idx.nonzero().flatten().tolist())
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        dataset = copy.copy(self)
        dataset.__indices__ = indices
        return dataset

    def shuffle(self, return_perm=False):
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __len__(self):
        if self.__indices__ is not None:
            return len(self.__indices__)
        else:
            return self.len()

    def __getitem__(self, index):
        if not isinstance(index, int):
            return self.index_select(index)
        else:
            return self.get(self.indices[index])

    def get(self, index: int):
        return self.data[index]

    def len(self):
        return len(self.data)


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
        current_index = len(self.data)
        subset_indices = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                input, output = line.split('\t')
                data_item = Text2TextDataItem(input_text=input, output_text=output, tokenizer=self.tokenizer,
                                              share_vocab=self.share_vocab)
                self.data[current_index] = data_item
                subset_indices.append(current_index)
                current_index += 1
        return subset_indices

    def build_vocab(self):
        data_for_vocab = [self.data[idx] for idx in self.split_ids['train']]
        if self.use_val_for_vocab:
            data_for_vocab += [self.data[idx] for idx in self.split_ids['val']]

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

    def vectorization(self):
        for item in self.data.values():
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

        torch.save(self.data, self.processed_file_paths['data'])

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
