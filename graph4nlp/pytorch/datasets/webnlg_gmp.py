import os
import torch

from ..data.dataset import Text2TextDataset, DataItem
from ..data.data import GraphData
import pickle as pkl
from multiprocessing import Pool
from copy import deepcopy
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
import numpy as np

class TripleGMP2TextDataItem(DataItem):
    def __init__(self, triples, gmp_seq, gmp_jump, output_text, rplc_dict, tokenizer, share_vocab=True):
        super(TripleGMP2TextDataItem, self).__init__(triples, tokenizer)
        self.triples = triples
        self.gmp_seq = gmp_seq
        self.gmp_jump = gmp_jump
        self.output_text = output_text
        self.rplc_dict = rplc_dict
        self.share_vocab = share_vocab

    def extract(self):
        """
        Returns
        -------
        Input tokens and output tokens
        """
        g: GraphData = self.graph

        input_tokens = []

        if self.tokenizer is None:
            input_tokens.extend(self.gmp_seq.strip().split(' '))
        else:
            input_tokens.extend(self.tokenizer(self.gmp_seq))

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

class WebNLGGMPDataset(Text2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train_gmp_data.pt', 'val': 'dev_gmp_data.pt', 'test': 'test_gmp_data.pt'}

    @property
    def processed_file_names(self):
        """At least 2 reserved keys should be fiiled: 'vocab' and 'data'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', max_word_vocab_size=None,
                 min_word_vocab_freq=1, word_emb_size=None, **kwargs):
        super(WebNLGGMPDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                              topology_subdir=topology_subdir, graph_type=graph_type,
                                              edge_strategy=edge_strategy, merge_strategy=merge_strategy,
                                              max_word_vocab_size=max_word_vocab_size,
                                              min_word_vocab_freq=min_word_vocab_freq,
                                              word_emb_size=word_emb_size, **kwargs)

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
        part = file_path.split('-')[0]
        with open(file_path, 'rb') as f:
            pt_data = pkl.load(f)

        for example in pt_data:
            triples = example['triples']
            gmp_seqs = example['gmp_seqs']
            gmp_jumps = example['gmp_jumps']
            output = example['text']
            rplc_dict = example['rplc']
            data_item = TripleGMP2TextDataItem(triples=triples,
                                               gmp_seq=gmp_seqs,
                                               gmp_jump=gmp_jumps,
                                               output_text=output,
                                               rplc_dict=rplc_dict,
                                               tokenizer=self.tokenizer,
                                               share_vocab=self.share_vocab)
            data.append(data_item)
        return data

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
    def triple_topology():
        return

    @staticmethod
    def _build_topology_process(data_items, topology_builder,
                                graph_type, dynamic_graph_type, dynamic_init_topology_builder,
                                merge_strategy, edge_strategy, dynamic_init_topology_aux_args,
                                lower_case, tokenizer, port, timeout):

        ret = []
        assert graph_type == 'static'

        pop_idxs = []
        for cnt, item in enumerate(data_items):
            if cnt % 1000 == 0:
                print("Port {}, processing: {} / {}".format(port, cnt, len(data_items)))

            graph = GraphData()
            triples = item.input_text.strip().split(' < TSP > ')
            nodes = []  # list of dicts
            nodes_tokens = []  # list
            edges = []  # list of lists
            for triple in triples:
                s, r, o = triple.split(' | ')

                # subject
                s_tokens = s.split()

                node1 = '_'.join(r.split())
                edge = 'A0'
                node2 = s_tokens[0]

                if node1 not in nodes_tokens:
                    nodes_tokens.append(node1)
                    nodes.append({'id': len(nodes), 'token': node1, 'type': 1})
                if node2 not in nodes_tokens:
                    nodes_tokens.append(node2)
                    nodes.append({'id': len(nodes), 'token': node2, 'type': 0})

                node1_id = nodes_tokens.index(node1)
                node2_id = nodes_tokens.index(node2)
                if [node1_id, edge, node2_id] not in edges:
                    edges.append([node1_id, edge, node2_id])

                # node type: subject->0, relation->1, object->2

                for s_token in s_tokens[1:]:
                    node1 = s_tokens[0]
                    edge = 'NE'
                    node2 = s_token

                    if node1 not in nodes_tokens:
                        nodes_tokens.append(node1)
                        nodes.append({'id': len(nodes), 'token': node1, 'type': 0})
                    if node2 not in nodes_tokens:
                        nodes_tokens.append(node2)
                        nodes.append({'id': len(nodes), 'token': node2, 'type': 0})

                    node1_id = nodes_tokens.index(node1)
                    node2_id = nodes_tokens.index(node2)
                    if [node1_id, edge, node2_id] not in edges:
                        edges.append([node1_id, edge, node2_id])

                # object
                o_tokens = o.split()

                node1 = '_'.join(r.split())
                edge = 'A1'
                node2 = o_tokens[0]

                if node1 not in nodes_tokens:
                    nodes_tokens.append(node1)
                    nodes.append({'id': len(nodes), 'token': node1, 'type': 1})
                if node2 not in nodes_tokens:
                    nodes_tokens.append(node2)
                    nodes.append({'id': len(nodes), 'token': node2, 'type': 2})

                node1_id = nodes_tokens.index(node1)
                node2_id = nodes_tokens.index(node2)
                if [node1_id, edge, node2_id] not in edges:
                    edges.append([node1_id, edge, node2_id])

                for o_token in o_tokens[1:]:
                    node1 = o_tokens[0]
                    edge = 'NE'
                    node2 = o_token

                    if node1 not in nodes_tokens:
                        nodes_tokens.append(node1)
                        nodes.append({'id': len(nodes), 'token': node1, 'type': 2})
                    if node2 not in nodes_tokens:
                        nodes_tokens.append(node2)
                        nodes.append({'id': len(nodes), 'token': node2, 'type': 2})

                    node1_id = nodes_tokens.index(node1)
                    node2_id = nodes_tokens.index(node2)
                    if [node1_id, edge, node2_id] not in edges:
                        edges.append([node1_id, edge, node2_id])

            node_num = len(nodes)
            graph.add_nodes(node_num)

            for node in nodes:
                graph.node_attributes[node['id']]['type'] = node['type']
                graph.node_attributes[node['id']]['token'] = node['token']

            for edge in edges:
                graph.add_edge(edge[0], edge[2])
                edge_idx = graph.edge_ids(edge[0], edge[2])[0]
                graph.edge_attributes[edge_idx]["token"] = edge[1]
            item.graph = graph

            ret.append(item)

        return ret

    def build_topology(self, data_items):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        total = len(data_items)
        thread_number = min(total, self.thread_number)

        data_items = self._build_topology_process(data_items, self.topology_builder, self.graph_type,
                                       self.dynamic_graph_type, self.dynamic_init_topology_builder,
                                       self.merge_strategy, self.edge_strategy, self.dynamic_init_topology_aux_args,
                                       self.lower_case, self.tokenizer, self.port, self.timeout)

        return data_items

    def vectorization(self, data_items):
        use_ie = False
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token, use_ie)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])

            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features['token_id'] = token_matrix

            tgt = item.output_text
            tgt_token_id = self.vocab_model.out_word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(self.vocab_model.out_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            item.output_np = tgt_token_id

            gmp_seq = item.gmp_seq
            gmp_seq_id = self.vocab_model.in_word_vocab.to_index_sequence(gmp_seq)
            gmp_seq_id.append(self.vocab_model.in_word_vocab.EOS)
            gmp_seq_id = np.array(gmp_seq_id)
            item.gmp_seq_np = gmp_seq_id

    @staticmethod
    def collate_fn(data_list: [TripleGMP2TextDataItem]):
        graph_data = [item.graph for item in data_list]
        from graph4nlp.pytorch.data.data import to_batch
        big_graph = to_batch(graph_data)

        # input_str = [deepcopy(item.input_text.strip()) for item in data_list]
        input_str = [item.input_text.strip() for item in data_list]

        gmp_jump = [item.gmp_jump for item in data_list]

        # output_numpy = [deepcopy(item.output_np) for item in data_list]
        output_numpy = [item.output_np for item in data_list]
        # output_str = [deepcopy(item.output_text.lower().strip()) for item in data_list]
        output_str = [item.output_text.strip() for item in data_list]
        output_pad = pad_2d_vals_no_size(output_numpy)
        tgt_seq = torch.from_numpy(output_pad).long()

        gmp_seq_numpy = [item.gmp_seq_np for item in data_list]
        gmp_seq_str = [item.gmp_seq.strip() for item in data_list]
        gmp_seq_pad = pad_2d_vals_no_size(gmp_seq_numpy)
        gmp_seq = torch.from_numpy(gmp_seq_pad).long()

        from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals
        max_num_tokens_a_node = max([x.graph.node_features['token_id'].size()[1] for x in data_list])
        if max_num_tokens_a_node > 1:
            for x in data_list:
                x.graph.node_features['token_id'] = torch.from_numpy(
                    pad_2d_vals(x.graph.node_features['token_id'].cpu().numpy(),
                                x.graph.node_features['token_id'].size()[0],
                                max_num_tokens_a_node)).long()

        def merge_jump(sequences):
            tmp = torch.ones(gmp_seq.size()).tolist()
            tmp_rev = []
            for i, seq in enumerate(sequences):
                for j in seq[:-1]:
                    tmp[i][j] = 0
                if len(seq)>0 and seq[-1] < len(tmp[0]):
                    tmp[i][seq[-1]] = 0
                tmp_rev.append(list(reversed(tmp[i])))

            return tmp, tmp_rev

        gmp_jump, gmp_jump_rev = merge_jump(gmp_jump)
        gmp_jump = torch.Tensor(gmp_jump)

        return {
            "graph_data": big_graph,
            "tgt_seq": tgt_seq,
            "input_str": input_str,
            "output_str": output_str,
            "gmp_seq": gmp_seq,
            "gmp_seq_str": gmp_seq_str,
            "gmp_jump": gmp_jump
        }