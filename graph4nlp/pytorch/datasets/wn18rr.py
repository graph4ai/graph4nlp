import os
import torch
import json
from nltk.tokenize import word_tokenize
from graph4nlp.pytorch.data.dataset import KGCompletionDataset, KGDataItem
# from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
# from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
# import numpy as np

dataset_root = '../test/dataset/WN18RR'


class WN18RRDataset(KGCompletionDataset):
    @property
    def raw_file_names(self) -> list:
        return ['e1rel_to_e2_train_small.json']

    @property
    def processed_file_names(self) -> dict:
        return {'vocab': 'vocab.pt', 'data': 'data.pt', 'KG_graph': 'KG_graph.pt'}

    def download(self):
        return
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, edge_strategy=None, **kwargs):
        self.split_token = '_'
        super(WN18RRDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                             topology_subdir=topology_subdir,edge_strategy=edge_strategy, **kwargs)
        self.data = torch.load(os.path.join(self.processed_dir, self.processed_file_names['data']))
        if 'KG_graph' in self.processed_file_names.keys():
            self.KG_graph = torch.load(os.path.join(self.processed_dir, self.processed_file_names['KG_graph']))
        self.build_vocab()


class WN18RRTestDataset(KGCompletionDataset):
    @property
    def raw_file_names(self) -> list:
        return ['e1rel_to_e2_ranking_test.json']

    @property
    def processed_file_names(self) -> dict:
        return {'vocab': 'test_vocab.pt', 'data': 'test_data.pt'}

    def download(self):
        return
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, edge_strategy=None, KG_graph=None, **kwargs):
        self.KG_graph = KG_graph
        self.split_token = '_'
        super(WN18RRTestDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                             topology_subdir=topology_subdir,edge_strategy=edge_strategy, **kwargs)
        self.data = torch.load(os.path.join(self.processed_dir, self.processed_file_names['data']))
        self.build_vocab()

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
                    if line_dict['e1'] not in self.graph_nodes:
                        continue

                    if line_dict['e2'] not in self.graph_nodes:
                        continue

                    data.append(KGDataItem(e1=line_dict['e1'],
                                           e2=line_dict['e2'],
                                           rel=line_dict['rel'],
                                           rel_eval=line_dict['rel_eval'],
                                           e2_multi=line_dict['e2_multi1'],
                                           e1_multi=line_dict['e2_multi2'],
                                           share_vocab=self.share_vocab,
                                           split_token=self.split_token))

        return data


    def build_topology(self):
        self.graph_nodes = self.KG_graph.graph_attributes['graph_nodes']
        self.graph_edges = self.KG_graph.graph_attributes['graph_edges']
        self.parsed_results = {}
        self.parsed_results['graph_content'] = []
        self.data: [KGDataItem] = self.build_dataitem(self.raw_file_paths)

        self.parsed_results['node_num'] = len(self.graph_nodes)
        self.parsed_results['graph_nodes'] = self.graph_nodes

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
            self.data[i].e1_tensor = torch.tensor(self.graph_nodes.index(e1), dtype=torch.long)

            e2 = self.data[i].e2
            self.data[i].e2_tensor = torch.tensor(self.graph_nodes.index(e2), dtype=torch.long)

            rel = self.data[i].rel
            if self.edge_strategy == "as_node":
                self.data[i].rel_tensor = torch.tensor(self.graph_nodes.index(rel), dtype=torch.long)
            else:
                self.data[i].rel_tensor = torch.tensor(self.graph_edges.index(rel), dtype=torch.long)

            rel_eval = self.data[i].rel_eval
            self.data[i].rel_eval_tensor = torch.tensor(self.graph_edges.index(rel_eval), dtype=torch.long)

            e2_multi = self.data[i].e2_multi
            self.data[i].e2_multi_tensor = torch.zeros(1, len(self.graph_nodes)).\
                scatter_(1, torch.tensor([self.graph_nodes.index(i)
                                          for i in e2_multi.split()
                                          if i in self.graph_nodes],
                                          dtype=torch.long).view(1, -1),
                         torch.ones(1, len(e2_multi.split()))).squeeze()
            self.data[i].e2_multi_tensor_idx = torch.tensor([self.graph_nodes.index(i)
                                                             for i in e2_multi.split()
                                                             if i in self.graph_nodes],
                                                            dtype=torch.long)

            e1_multi = self.data[i].e1_multi
            self.data[i].e1_multi_tensor = torch.zeros(1, len(self.graph_nodes)). \
                scatter_(1, torch.tensor([self.graph_nodes.index(i)
                                          for i in e1_multi.split()
                                          if i in self.graph_nodes],
                                         dtype=torch.long).view(1, -1),
                         torch.ones(1, len(e1_multi.split()))).squeeze()
            self.data[i].e1_multi_tensor_idx = torch.tensor([self.graph_nodes.index(i)
                                                             for i in e1_multi.split()
                                                             if i in self.graph_nodes],
                                                             dtype=torch.long)


        torch.save(self.data, os.path.join(self.processed_dir, self.processed_file_names['data']))
        if 'KG_graph' in self.processed_file_names.keys():
            torch.save(self.KG_graph, os.path.join(self.processed_dir, self.processed_file_names['KG_graph']))

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


if __name__ == '__main__':
    wn18rr_dataset = WN18RRDataset(root_dir='/Users/gaohanning/PycharmProjects/graph4nlp/examples/pytorch/kg_completion/WN18RR',
                                  topology_builder=None,
                                  topology_subdir='e1rel_to_e2')

    wn18rr_test_dataset = WN18RRTestDataset(
        root_dir='/Users/gaohanning/PycharmProjects/graph4nlp/examples/pytorch/kg_completion/WN18RR',
        KG_graph=wn18rr_dataset.KG_graph,
        topology_builder=None,
        topology_subdir='e1rel_to_e2')

    a = 0
