import os
import torch
import json
from nltk.tokenize import word_tokenize
from graph4nlp.pytorch.data.dataset import KGCompletionDataset, KGDataItem
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from ..data.data import GraphData
from ..modules.utils.vocab_utils import Vocab
import numpy as np

dataset_root = '../test/dataset/kinship'



class KinshipDataset(KGCompletionDataset):
    @property
    def raw_file_names(self) -> list:
        return ['e1rel_to_e2_train.json']
        # return ['sequence.txt']

    @property
    def processed_file_names(self) -> dict:
        return {'vocab': 'vocab.pt', 'data': 'data.pt', 'KG_graph': 'KG_graph.pt'}

    def download(self):
        return
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, edge_strategy=None, **kwargs):
        super(KinshipDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                             topology_subdir=topology_subdir,edge_strategy=edge_strategy, **kwargs)
        self.data = torch.load(os.path.join(self.processed_dir, self.processed_file_names['data']))
        if 'KG_graph' in self.processed_file_names.keys():
            self.KG_graph = torch.load(os.path.join(self.processed_dir, self.processed_file_names['KG_graph']))
        self.build_vocab()


class KinshipTestDataset(KGCompletionDataset):
    @property
    def raw_file_names(self) -> list:
        return ['e1rel_to_e2_ranking_test.json']
        # return ['sequence.txt']

    @property
    def processed_file_names(self) -> dict:
        return {'vocab': 'test_vocab.pt', 'data': 'test_data.pt'}

    def download(self):
        return
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, edge_strategy=None, KG_graph=None, **kwargs):
        self.KG_graph = KG_graph
        super(KinshipTestDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                             topology_subdir=topology_subdir,edge_strategy=edge_strategy, **kwargs)
        self.data = torch.load(os.path.join(self.processed_dir, self.processed_file_names['data']))
        self.build_vocab()

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
            # self.data[i].e1_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e1), dtype=torch.long)
            self.data[i].e1_tensor = torch.tensor(self.graph_nodes.index(e1), dtype=torch.long)

            e2 = self.data[i].e2
            # self.data[i].e2_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e2),
            #                                       dtype=torch.long)
            self.data[i].e2_tensor = torch.tensor(self.graph_nodes.index(e2), dtype=torch.long)

            rel = self.data[i].rel
            # self.data[i].rel_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(rel),
            #                                        dtype=torch.long)
            if self.edge_strategy == "as_node":
                self.data[i].rel_tensor = torch.tensor(self.graph_nodes.index(rel), dtype=torch.long)
            else:
                self.data[i].rel_tensor = torch.tensor(self.graph_edges.index(rel), dtype=torch.long)

            rel_eval = self.data[i].rel_eval
            # self.data[i].rel_eval_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(rel_eval),
            #                                             dtype=torch.long)
            self.data[i].rel_eval_tensor = torch.tensor(self.graph_edges.index(rel_eval), dtype=torch.long)

            e2_multi = self.data[i].e2_multi
            # self.data[i].e2_multi_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e2_multi),
            #                                       dtype=torch.long)
            self.data[i].e2_multi_tensor = torch.zeros(1, len(self.graph_nodes)).\
                scatter_(1,
                         torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long).view(1, -1),
                         torch.ones(1, len(e2_multi.split()))).squeeze()
            self.data[i].e2_multi_tensor_idx = torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long)
            # self.data[i].e2_multi_tensor = torch.tensor([self.graph_nodes.index(i) for i in e2_multi.split()], dtype=torch.long)

            e1_multi = self.data[i].e1_multi
            # self.data[i].e1_multi_tensor = torch.tensor(self.vocab_model.in_word_vocab.to_index_sequence(e1_multi),
            #                                             dtype=torch.long)
            self.data[i].e1_multi_tensor = torch.zeros(1, len(self.graph_nodes)). \
                scatter_(1,
                         torch.tensor([self.graph_nodes.index(i) for i in e1_multi.split()], dtype=torch.long).view(1, -1),
                         torch.ones(1, len(e1_multi.split()))).squeeze()
            self.data[i].e1_multi_tensor_idx = torch.tensor([self.graph_nodes.index(i) for i in e1_multi.split()],
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
    kinshipdataset = KinshipDataset(root_dir='../test/dataset/kinship', topology_builder=DependencyBasedGraphConstruction,
                topology_subdir='e1rel_to_e2')

    a = 0
