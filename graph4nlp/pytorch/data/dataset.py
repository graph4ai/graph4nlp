import abc
import copy
import os

import numpy as np
import stanfordcorenlp
import torch.utils.data
from nltk.tokenize import word_tokenize

from ..data.data import GraphData
from ..modules.utils.vocab_utils import VocabModel, Vocab

import json
from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction

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


class KGDataItem(DataItem):
    def __init__(self, e1, rel, e2, rel_eval, e2_multi, e1_multi, share_vocab=True):
        super(KGDataItem, self).__init__(input_text=None, tokenizer=None)
        self.e1 = e1
        self.rel = rel
        self.e2 = e2
        self.rel_eval = rel_eval
        self.e2_multi = e2_multi
        self.e1_multi = e1_multi
        self.share_vocab = share_vocab

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
            rel_tokens = self.rel.strip().split(' ')
        else:
            rel_tokens = self.tokenizer(self.rel)

        if self.tokenizer is None:
            rel_eval_tokens = self.rel_eval.strip().split(' ')
        else:
            rel_eval_tokens = self.tokenizer(self.rel_eval)

        if self.share_vocab:
            return e1_tokens + e2_tokens + e1_multi_tokens + e2_multi_tokens + rel_tokens + rel_eval_tokens
        else:
            return e1_tokens + e2_tokens + e1_multi_tokens + e2_multi_tokens, rel_tokens + rel_eval_tokens


class KGDataset(Dataset):
    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, share_vocab=True,
                 edge_strategy=None, **kwargs):
        self.data_item_type = KGDataItem
        self.share_vocab = share_vocab  # share vocab between entity and relation
        # self.KG_graph = GraphData()
        self.edge_strategy = edge_strategy
        super(KGDataset, self).__init__(root_dir, topology_builder, topology_subdir, **kwargs)

    def build_vocab(self):
        vocab_model = VocabModel.build(saved_vocab_file=os.path.join(self.processed_dir, self.processed_file_names['vocab']),
                               data_set=self.data,
                               tokenizer=self.tokenizer,
                               lower_case=self.lower_case,
                               max_word_vocab_size=None,
                               min_word_vocab_freq=1,
                               pretrained_word_emb_file=self.pretrained_word_emb_file,
                               word_emb_size=300)
        self.vocab_model = vocab_model

        return self.vocab_model

    def build_dataitem(self, files):
        r"""Read raw input file and build DataItem out of it.
        The format of the input file should contain lines of input, each line representing one record of data.
        The input and output is seperated by a tab(\t).

        Examples
        --------
        {"e1": "person84", "e2": "person85", "rel": "term21", "rel_eval": "term21_reverse",
        "e2_multi1": "person85",
        "e2_multi2": "person74 person84 person55 person96 person66 person57"}

        {"e1": "person20", "e2": "person90", "rel": "term11", "rel_eval": "term11_reverse",
        "e2_multi1": "person29 person82 person85 person77 person73 person63 person34 person86 person4
        person83 person46 person16 person48 person17 person59 person80 person50 person90",
        "e2_multi2": "person29 person82 person2 person20 person83 person46 person80"}

        Parameters
        ----------
        file: list of json files
            The list containing all the input file paths.
        """
        data = []
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_dict = json.loads(line)
                    data.append(KGDataItem(e1=line_dict['e1'],
                                           e2=line_dict['e2'],
                                           rel=line_dict['rel'],
                                           rel_eval=line_dict['rel_eval'],
                                           e2_multi=line_dict['e2_multi1'],
                                           e1_multi=line_dict['e2_multi2'],
                                           share_vocab=self.share_vocab))

                    for e2 in line_dict['e2_multi1'].split(' '):
                        triple = [line_dict['e1'], line_dict['rel'], e2]

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
                            if triple not in self.parsed_results['graph_content']:
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

        return data

    def build_topology(self):
        self.graph_nodes = []
        self.graph_edges = []
        self.parsed_results = {}
        self.parsed_results['graph_content'] = []
        self.data: [KGDataItem] = self.build_dataitem(self.raw_file_paths)

        self.KG_graph = GraphData()
        self.parsed_results['node_num'] = len(self.graph_nodes)
        self.parsed_results['graph_nodes'] = self.graph_nodes
        self.KG_graph = IEBasedGraphConstruction._construct_static_graph(self.parsed_results, edge_strategy=self.edge_strategy)
        self.KG_graph.graph_attributes['num_entities'] = len(self.graph_nodes)
        self.KG_graph.graph_attributes['num_relations'] = len(self.graph_edges)
        self.KG_graph.graph_attributes['graph_nodes'] = self.graph_nodes
        self.KG_graph.graph_attributes['graph_edges'] = self.graph_edges

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths]):
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        self.build_topology()
        self.build_vocab()
        self.vectorization()

    def vectorization(self):
        graph: GraphData = self.KG_graph
        token_matrix = []
        for node_idx in range(graph.get_node_num()):
            node_token = graph.node_attributes[node_idx]['token']
            node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token)
            graph.node_attributes[node_idx]['token_id'] = node_token_id
            token_matrix.append([node_token_id])
        token_matrix = torch.tensor(token_matrix, dtype=torch.long)
        graph.node_features['token_id'] = token_matrix

        for i in range(len(self.data)):
            e1 = self.data[i].e1
            # self.data[i].e1_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e1), dtype=torch.long)
            self.data[i].e1_tensor = torch.tensor(self.graph_nodes.index(e1), dtype=torch.long)

            # e2 = self.data[i].e2
            # self.data[i].e2_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e2),
            #                                       dtype=torch.long)
            # self.data[i].e2_tensor = torch.tensor(self.graph_nodes.index(e2), dtype=torch.long)

            rel = self.data[i].rel
            # self.data[i].rel_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(rel),
            #                                        dtype=torch.long)
            if self.edge_strategy == "as_node":
                self.data[i].rel_tensor = torch.tensor(self.graph_nodes.index(rel), dtype=torch.long)
            else:
                self.data[i].rel_tensor = torch.tensor(self.graph_edges.index(rel), dtype=torch.long)

            # rel_eval = self.data[i].rel_eval
            # self.data[i].rel_eval_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(rel_eval),
            #                                             dtype=torch.long)
            # self.data[i].rel_eval_tensor = torch.tensor(self.graph_nodes.index(rel_eval), dtype=torch.long)

            e2_multi = self.data[i].e2_multi
            # self.data[i].e2_multi_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e2_multi),
            #                                       dtype=torch.long)
            self.data[i].e2_multi_tensor = torch.zeros(1, len(self.graph_nodes)).\
                scatter_(1,
                         torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long).view(1, -1),
                         torch.ones(1, len(e2_multi.split()))).squeeze()
            self.data[i].e2_multi_tensor_idx = torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long)
            # self.data[i].e2_multi_tensor = torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long)

            # e1_multi = self.data[i].e1_multi
            # self.data[i].e1_multi_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e1_multi),
            #                                             dtype=torch.long)
            # self.data[i].e1_multi_tensor = torch.tensor(e1_multi, dtype=torch.long)


        torch.save(self.data, os.path.join(self.processed_dir, self.processed_file_names['data']))
        if 'KG_graph' in self.processed_file_names.keys():
            torch.save(self.KG_graph, os.path.join(self.processed_dir, self.processed_file_names['KG_graph']))

    @staticmethod
    def collate_fn(data_list: [KGDataItem]):
        # graph_data = [item.graph for item in data_list]
        e1 = torch.tensor([item.e1_tensor for item in data_list])
        rel = torch.tensor([item.rel_tensor for item in data_list])
        # e2_multi_tensor_idx = torch.tensor([item.e2_multi_tensor_idx for item in data_list])

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

        # do padding here
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

        # # do padding here
        # seq_len = [item.output_tensor.shape[0] for item in data_list]
        # max_seq_len = max(seq_len)
        # tgt_seq_pad = []
        # for item in data_list:
        #     if item.output_tensor.shape[0] < max_seq_len:
        #         need_pad_length = max_seq_len - item.output_tensor.shape[0]
        #         pad = torch.zeros(need_pad_length).fill_(Vocab.PAD)
        #         tgt_seq_pad.append(torch.cat((item.output_tensor, pad.long()), dim=0).unsqueeze(0))
        #     elif item.output_tensor.shape[0] == max_seq_len:
        #         tgt_seq_pad.append(item.output_tensor.unsqueeze(0))
        #     else:
        #         raise RuntimeError("Size mismatch error")
        #
        # tgt_seq = torch.cat(tgt_seq_pad, dim=0)
        return [e1, rel, e2_multi, e2_multi_tensor_idx]