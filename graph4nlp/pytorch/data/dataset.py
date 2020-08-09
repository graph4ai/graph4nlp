import abc
import copy
import os

import numpy as np
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

    Parameters
    ----------
    root: str
        The root directory path where the dataset is stored.
    """

    @property
    def raw_file_names(self) -> list:
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
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def build_dataitem(self):
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
                **kwargs):
        super(Dataset, self).__init__()

        self.root = root
        self.tokenizer = tokenizer
        self.lower_case = lower_case
        self.pretrained_word_emb_file = pretrained_word_emb_file
        self.topology_builder = topology_builder
        self.topology_subdir = topology_subdir
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
    def raw_file_paths(self) -> list:
        """The paths to raw files."""
        return [os.path.join(self.raw_dir, raw_file_name) for raw_file_name in self.raw_file_names]

    @property
    def processed_file_paths(self) -> list:
        return [os.path.join(self.processed_dir, processed_file_name) for processed_file_name in
                self.processed_file_names.values()]

    def _download(self):
        if all([os.path.exists(raw_path) for raw_path in self.raw_file_paths]):
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def build_topology(self):
        self.data: [Text2TextDataItem] = self.build_dataitem(self.raw_file_paths)

        if self.graph_type == 'static':
            processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)
            for item in self.data:
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
        vocab_model = VocabModel.build(saved_vocab_file=os.path.join(self.processed_dir, 'vocab.pt'),
                               data_set=self.data,
                               tokenizer=self.tokenizer,
                               lower_case=self.lower_case,
                               max_word_vocab_size=None,
                               min_word_vocab_freq=1,
                               pretrained_word_emb_file=self.pretrained_word_emb_file,
                               word_emb_size=300)
        self.vocab_model = vocab_model

        return self.vocab_model

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths]):
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        self.build_topology()
        self.build_vocab()
        self.vectorization()

    def indices(self):
        if self.__indices__ is not None:
            return self.__indices__
        else:
            return range(len(self))

    def index_select(self, idx):
        indices = self.indices()

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
            return self.get(self.indices()[index])

    def get(self, index: int):
        return self.data[index]

    def len(self):
        return len(self.data)


class TextToTextDataset(Dataset):
    def __init__(self, root_dir, topology_builder, topology_subdir, share_vocab=True, **kwargs):
        self.data_item_type = Text2TextDataItem
        self.share_vocab = share_vocab
        super(TextToTextDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def build_dataitem(self, files):
        r"""Read raw input file and build DataItem out of it.
        The format of the input file should contain lines of input, each line representing one record of data.
        The input and output is seperated by a tab(\t).

        Examples
        --------
        list job use languageid0	job ( ANS ) , language ( ANS , languageid0 )
        show job use languageid0	job ( ANS ) , language ( ANS , languageid0 )

        In the above the input is the natural language in the left ("list job use languageid0") and the output
        is in the right.

        Parameters
        ----------
        file: list
            The list containing all the input file paths.
        """
        data = []
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    input, output = line.split('\t')
                    data.append(Text2TextDataItem(input_text=input, output_text=output, tokenizer=self.tokenizer,
                                                  share_vocab=self.share_vocab))
        return data

    def build_vocab(self):
        vocab_model = VocabModel.build(saved_vocab_file=os.path.join(self.processed_dir, 'vocab.pt'),
                                       data_set=self.data,
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
        for i in range(len(self.data)):
            graph: GraphData = self.data[i].graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features['token_id'] = token_matrix

            tgt = self.data[i].output_text
            tgt_token_id = self.vocab_model.in_word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(self.vocab_model.in_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            tgt_token_id = torch.from_numpy(tgt_token_id)
            self.data[i].output_tensor = tgt_token_id

        torch.save(self.data, os.path.join(self.processed_dir, self.processed_file_names['data']))

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
