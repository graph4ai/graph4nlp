# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import re
import io
import random
import string
# from nltk.tokenize import wordpunct_tokenize
from collections import Counter, defaultdict, OrderedDict

import numpy as np
from scipy.sparse import *
import torch

from graph4nlp.pytorch.test.seq_decoder.graph2seq.src.g2s_v2.core.utils.timer import Timer
from . import padding_utils
from . import constants

tokenize = lambda s: re.split("\\s+", s)


def vectorize_input(batch, config, training=True, device=None):
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    # batch_size = len(batch.sent1_word)
    batch_size = batch.batch_size

    in_graphs = {}
    for k, v in batch.sent1_graph.items():
        if k in ['node2edge', 'edge2node', 'max_num_edges']:
            in_graphs[k] = v
        else:
            in_graphs[k] = torch.LongTensor(v).to(device) if device else torch.LongTensor(v)

    questions = torch.LongTensor(batch.sent2_word)
    question_lens = torch.LongTensor(batch.sent2_length)

    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'in_graphs': in_graphs,
                   'targets': questions.to(device) if device else questions,
                   'target_lens': question_lens.to(device) if device else question_lens,
                   'target_src': batch.sent2_src,
                   'oov_dict': batch.oov_dict}

        return example

def prepare_datasets(config):
    if config['trainset'] is not None:
        train_set, train_seq_lens = load_data(config['trainset'], isLower=config['dataset_lower'])
        # train_set, train_seq_lens = load_data(config['trainset'], isLower=True)
        print('# of training examples: {}'.format(len(train_set)))
        print('[ Max training seq length: {} ]'.format(np.max(train_seq_lens)))
        print('[ Min training seq length: {} ]'.format(np.min(train_seq_lens)))
        print('[ Mean training seq length: {} ]'.format(int(np.mean(train_seq_lens))))
    else:
        train_set = None

    if config['devset'] is not None:
        dev_set, dev_seq_lens = load_data(config['devset'], isLower=True)
        print('# of dev examples: {}'.format(len(dev_set)))
        print('[ Max dev seq length: {} ]'.format(np.max(dev_seq_lens)))
        print('[ Min dev seq length: {} ]'.format(np.min(dev_seq_lens)))
        print('[ Mean dev seq length: {} ]'.format(int(np.mean(dev_seq_lens))))
    else:
        dev_set = None

    if config['testset'] is not None:
        test_set, test_seq_lens = load_data(config['testset'], isLower=True)
        print('# of testing examples: {}'.format(len(test_set)))
        print('[ Max testing seq length: {} ]'.format(np.max(test_seq_lens)))
        print('[ Min testing seq length: {} ]'.format(np.min(test_seq_lens)))
        print('[ Mean testing seq length: {} ]'.format(int(np.mean(test_seq_lens))))
    else:
        test_set = None

    if config.get('sample_example', None):
        a, b, c = config['sample_example'].split(';')

        if train_set is not None:
            np.random.shuffle(train_set)
            n_train = int(len(train_set) * float(a))
            train_set = train_set[:n_train]
            print('Sampled {} training examples'.format(n_train))

        if dev_set is not None:
            np.random.shuffle(dev_set)
            n_dev = int(len(dev_set) * float(b))
            dev_set = dev_set[:n_dev]
            print('Sampled {} dev examples'.format(n_dev))

        if test_set is not None:
            np.random.shuffle(test_set)
            n_test = int(len(test_set) * float(c))
            test_set = test_set[:n_test]
            print('Sampled {} testing examples'.format(n_test))

    return {'train': train_set, 'dev': dev_set, 'test': test_set}


def load_data(inpath, isLower=True):
    all_instances = []
    all_seq_lens = []

    with open(inpath, 'r') as f:
        for line in f:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            seq = jo['seq']

            graph = {}
            # graph['g_ids'] = jo['g_ids']
            # graph['g_ids_features'] = jo['g_ids_features']
            # graph['g_adj'] = jo['g_adj']

            graph['g_features'] = [jo['g_ids_features'][str(i)] for i in range(len(jo['g_ids_features']))]
            graph['g_adj'] = defaultdict(set)
            num_edges = 0
            for k, v in jo['g_adj'].items():
                for each in v:
                    graph['g_adj'][int(k)].add(int(each))
                    num_edges += 1

            # # add word seq edges to graph
            # for i in range(len(jo['g_ids_features']) - 1):
            #     if not i + 1 in graph['g_adj'][i]:
            #         graph['g_adj'][i].add(i + 1)
            #         num_edges += 1

            #     if not i in graph['g_adj'][i + 1]:
            #         graph['g_adj'][i + 1].add(i)
            #         num_edges += 1

            graph['num_edges'] = num_edges

            all_instances.append([Sequence(graph, is_graph=True, isLower=isLower), Sequence(seq, isLower=isLower, end_sym=constants._EOS_TOKEN)])
            all_seq_lens.append(len(all_instances[-1][1].words))
    return all_instances, all_seq_lens



class DataStream(object):
    def __init__(self, all_instances, word_vocab, edge_vocab, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1, ext_vocab=False):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda instance: len(instance[0].graph['g_features'])) # the last element is label
        else:
            np.random.shuffle(all_instances)
            np.random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = InstanceBatch(cur_instances, config, word_vocab, edge_vocab, ext_vocab=ext_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]

class InstanceBatch(object):
    def __init__(self, instances, config, word_vocab, edge_vocab, ext_vocab=False):
        self.instances = instances
        self.batch_size = len(instances)
        self.oov_dict = None # out-of-vocabulary dict

        # Create word representation and length
        self.sent1_word = [] # [batch_size, graph_size, node_attr_len]
        self.sent1_length = [] # [batch_size]
        self.sent1_node_length = [] # [batch_size, graph_size]

        self.sent2_src = []
        self.sent2_word = [] # [batch_size, sent2_len]
        self.sent2_length = [] # [batch_size]


        if ext_vocab:
            base_oov_idx = len(word_vocab)
            self.oov_dict = OOVDict(base_oov_idx)


        # Build graph
        batch_graphs = [each[0].graph for each in instances]
        print("ooooooo")
        self.sent1_graph = vectorize_batch_graph(batch_graphs, word_vocab, self.oov_dict, ext_vocab=ext_vocab, copy_node=config.get('copy_node', False))



        for i, (sent1, sent2) in enumerate(instances):
            if ext_vocab and config.get('copy_node', False):
                sent2_idx = seq2ext_vocab_id(i, sent2.words, word_vocab, self.oov_dict)
            else:
                sent2_idx = []
                for word in sent2.words:
                    idx = word_vocab.getIndex(word)
                    if ext_vocab and idx == word_vocab.UNK:
                        idx = self.oov_dict.word2index.get((i, word), idx)
                    sent2_idx.append(idx)

            self.sent2_word.append(sent2_idx)
            self.sent2_src.append(sent2.src)
            self.sent2_length.append(len(sent2.words))

        self.sent2_word = padding_utils.pad_2d_vals_no_size(self.sent2_word)
        self.sent2_length = np.array(self.sent2_length, dtype=np.int32)



class Sequence(object):
    def __init__(self, data, is_graph=False, isLower=False, end_sym=None):
        self.graph = data if is_graph else None
        if is_graph:
            if isLower:
                self.graph['g_features'] = [tokenize(each.lower()) for each in self.graph['g_features']]
            else:
                self.graph['g_features'] = [tokenize(each) for each in self.graph['g_features']]

        else:
            self.src = ' '.join(tokenize(data))
            self.tokText = data

            # it's the output sequence
            if end_sym != None:
                self.tokText += ' ' + end_sym

            if isLower:
                self.src = self.src.lower()
                self.tokText = self.tokText.lower()

            self.words = tokenize(self.tokText)



def vectorize_batch_graph(graphs, word_vocab, oov_dict, ext_vocab=False, copy_node=False):
    num_nodes = max([len(g['g_features']) for g in graphs])
    num_edges = max([g['num_edges'] for g in graphs])

    batch_node_feat = []
    batch_node_lens = []
    batch_num_nodes = []
    batch_node2edge = []
    batch_edge2node = []

    if ext_vocab and copy_node:
        batch_g_oov_idx = []



    for example_id, g in enumerate(graphs):
        # Node names
        node_feat_idx = []
        if ext_vocab and copy_node:
            g_oov_idx = []

        for each in g['g_features']: # node level
            # Add out of vocab
            if ext_vocab and copy_node:
                oov_idx = oov_dict.add_word(example_id, tuple(each))
                g_oov_idx.append(oov_idx)

            tmp_node_idx = []
            for word in each:
                idx = word_vocab.getIndex(word)
                if not copy_node and ext_vocab and idx == word_vocab.UNK:
                    idx = oov_dict.add_word(example_id, word)
                tmp_node_idx.append(idx)

            node_feat_idx.append(tmp_node_idx)

        batch_node_feat.append(node_feat_idx)
        batch_node_lens.append([max(len(x), 1) for x in node_feat_idx])
        batch_num_nodes.append(len(node_feat_idx))

        if ext_vocab and copy_node:
            batch_g_oov_idx.append(g_oov_idx)
            assert len(g_oov_idx) == len(node_feat_idx)

        node2edge = lil_matrix(np.zeros((num_edges, num_nodes)), dtype=np.float32)
        edge2node = lil_matrix(np.zeros((num_nodes, num_edges)), dtype=np.float32)
        edge_index = 0
        for node1, value in g['g_adj'].items():
            node1 = int(node1)
            for each in value:
                node2 = int(each)
                if node1 == node2: # Ignore self-loops for now
                    continue
                node2edge[edge_index, node2] = 1
                edge2node[node1, edge_index] = 1
                edge_index += 1
        batch_node2edge.append(node2edge)
        batch_edge2node.append(edge2node)


    batch_node_feat = padding_utils.pad_3d_vals_no_size(batch_node_feat, fills=word_vocab.PAD)
    batch_node_lens = padding_utils.pad_2d_vals_no_size(batch_node_lens, fills=1)
    batch_num_nodes = np.array(batch_num_nodes, dtype=np.int32)

    batch_graphs = {'max_num_edges': num_edges,
                    'node_feats': batch_node_feat,
                    'node_lens': batch_node_lens,
                    'num_nodes': batch_num_nodes,
                    'node2edge': batch_node2edge,
                    'edge2node': batch_edge2node
                    }

    if ext_vocab and copy_node:
        batch_g_oov_idx = padding_utils.pad_2d_vals_no_size(batch_g_oov_idx, fills=word_vocab.PAD)
        batch_graphs['g_oov_idx'] = batch_g_oov_idx

    return batch_graphs


class OOVDict(object):
    def __init__(self, base_oov_idx):
        self.word2index = {}  # type: Dict[Tuple[int, str], int]
        self.index2word = {}  # type: Dict[Tuple[int, int], str]
        self.next_index = {}  # type: Dict[int, int]
        self.base_oov_idx = base_oov_idx
        self.ext_vocab_size = base_oov_idx

    def add_word(self, idx_in_batch, word) -> int:
        key = (idx_in_batch, word)
        index = self.word2index.get(key, None)
        if index is not None: return index
        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
        return index



def find_sublist(src_list, a_list):
    indices = []
    for i in range(len(src_list)):
        if src_list[i: i + len(a_list)] == a_list:
            start_idx = i
            end_idx = i + len(a_list)
            indices.append((start_idx, end_idx))
    return indices

def seq2ext_vocab_id(idx_in_batch, seq, word_vocab, oov_dict):
    matched_pos = {}
    for key in oov_dict.word2index:
        if key[0] == idx_in_batch:
            indices = find_sublist(seq, list(key[1]))
            for pos in indices:
                matched_pos[pos] = key

    matched_pos = sorted(matched_pos.items(), key=lambda d: d[0][0])

    seq_idx = []
    i = 0
    while i < len(seq):
        if len(matched_pos) == 0 or i < matched_pos[0][0][0]:
            seq_idx.append(word_vocab.getIndex(seq[i]))
            i += 1
        else:
            pos, key = matched_pos.pop(0)
            seq_idx.append(oov_dict.word2index.get(key))
            i += len(key[1])
    return seq_idx
