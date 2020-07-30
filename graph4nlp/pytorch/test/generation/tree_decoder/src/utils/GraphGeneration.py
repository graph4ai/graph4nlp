import argparse
import copy
import json
import os
import pickle as pkl
import random
import time
from collections import OrderedDict

import networkx as nx
import networkx.algorithms as nxalg
import numpy as np
import torch
import tqdm
from pythonds.basic.stack import Stack
from stanfordcorenlp import StanfordCoreNLP

from .data_utils import SymbolsManager, convert_to_tree


class Node():
    def __init__(self, word, type_, id_):
        # word: this node's text
        self.word = word

        # type: 0 for word nodes, 1 for constituency nodes, 2 for dependency nodes(if they exists)
        self.type = type_

        # id: unique identifier for every node
        self.id = id_

        self.head = False

        self.tail = False

    def __str__(self):
        return self.word


class GraphGenerator():

    def __init__(self, args):
        self.args = args
        self.seed = args.seed

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.source_data_dir = args.source_data_dir
        self.output_data_dir = args.output_data_dir
        self.batch_size = args.batch_size
        self.min_freq = args.min_freq
        self.max_vocab_size = args.max_vocab_size

        if args.parse_chinese_or_english == 0:
            self.parser = StanfordCoreNLP('http://localhost', port=9000)
        elif args.parse_chinese_or_english == 1:
            self.parser = StanfordCoreNLP('http://localhost', port=9001)

        print("syntactic parser ready..\n")

    def parse(self, src_text):
        nlp_parser = self.parser
        output = nlp_parser.annotate(src_text.strip(), properties={
                        'annotators': "tokenize,ssplit,pos,parse",
                        "tokenize.options":"splitHyphenated=true,normalizeParentheses=false",
        	        	"tokenize.whitespace": True,
                        'ssplit.isOneSentence': True,
                        'outputFormat': 'json'
                    })
        output = json.loads(output)['sentences'][0]
        word_list = [x['originalText'] for x in output['tokens']]
        pos_list = [(x['originalText'], x['pos']) for x in output['tokens']]
        dep_list = [(x['governor']-1, x['dependent']-1) for x in output['basicDependencies']][1:]
        parse_uni_str = output['parse']

        return parse_uni_str

    ''' Some data processing func '''

    def split_str(self, string):
        dele_char = '\uff0e'
        if dele_char not in string:
            return [string]
        else:
            s_arr = string.split(dele_char)
            res = []
            for index in range(len(s_arr)):
                if len(s_arr[index]) > 0:
                    if index != len(s_arr)-1:
                        res.append(s_arr[index] + dele_char)
                    else:
                        res.append(s_arr[index])
            return res

    def cut_root_node(self, con_string):
        tmp = con_string
        if con_string[0] == '(' and con_string[-1] == ')':
            tmp = con_string[1:-1].replace("ROOT", "")
            if tmp[0] == '\n':
                tmp = tmp[1:]
        return tmp

    def cut_pos_node(self, g):
        node_arr = list(g.nodes())
        del_arr = []
        for n in node_arr:
            edge_arr = list(g.edges())
            cnt_in = 0
            cnt_out = 0
            for e in edge_arr:
                if n.id == e[0].id:
                    cnt_out += 1
                    out_ = e[1]
                if n.id == e[1].id:
                    cnt_in += 1
                    in_ = e[0]
            if cnt_in == 1 and cnt_out == 1 and out_.type == 0:
                del_arr.append((n, in_, out_))
        for d in del_arr:
            g.remove_node(d[0])
            g.add_edge(d[1], d[2])
        return g

    def cut_line_node(self, g):
        node_arr = list(g.nodes())

        for n in node_arr:
            edge_arr = list(g.edges())
            cnt_in = 0
            cnt_out = 0
            for e in edge_arr:
                if n.id == e[0].id:
                    cnt_out += 1
                    out_ = e[1]
                if n.id == e[1].id:
                    cnt_in += 1
                    in_ = e[0]
            if cnt_in == 1 and cnt_out == 1:
                g.remove_node(n)
                g.add_edge(in_, out_)
        return g
    '''Some funcs in cons tree'''

    def get_seq_nodes(self, g):
        res = []
        node_arr = list(g.nodes())
        for n in node_arr:
            if n.type == 0:
                res.append(copy.deepcopy(n))
        return sorted(res, key=lambda x: x.id)

    def get_non_seq_nodes(self, g):
        res = []
        node_arr = list(g.nodes())
        for n in node_arr:
            if n.type != 0:
                res.append(copy.deepcopy(n))
        return sorted(res, key=lambda x: x.id)

    def get_all_text(self, g):
        seq_arr = self.get_seq_nodes(g)
        nonseq_arr = self.get_non_seq_nodes(g)
        seq = [x.word for x in seq_arr]
        nonseq = [x.word for x in nonseq_arr]
        return seq + nonseq

    def get_all_id(self, g):
        seq_arr = self.get_seq_nodes(g)
        nonseq_arr = self.get_non_seq_nodes(g)
        seq = [x.id for x in seq_arr]
        nonseq = [x.id for x in nonseq_arr]
        return seq + nonseq

    def get_id2word(self, g):
        res = {}
        seq_arr = self.get_seq_nodes(g)
        nonseq_arr = self.get_non_seq_nodes(g)
        for x in seq_arr:
            res[x.id] = x.word
        for x in nonseq_arr:
            res[x.id] = x.word
        return res

    def nodes_to_string(self, l):
        return " ".join([x.word for x in l])

    def print_edges(self, g):
        edge_arr = list(g.edges())
        for e in edge_arr:
            print((e[0].word, e[1].word), (e[0].id, e[1].id))

    def print_nodes(self, g, he_ta=False):
        nodes_arr = list(g.nodes())
        if he_ta:
            print([(n.word, n.id, n.head, n.tail) for n in nodes_arr])
        else:
            print([(n.word, n.id) for n in nodes_arr])

    def graph_connect(self, a_, b_):
        a = copy.deepcopy(a_)
        b = copy.deepcopy(b_)
        max_id = 0
        for n in a.nodes():
            if n.id > max_id:
                max_id = n.id
        tmp = copy.deepcopy(b)
        for n in tmp.nodes():
            n.id += max_id

        res = nx.union(a, tmp)
        seq_nodes_arr = []
        for n in res.nodes():
            if n.type == 0:
                seq_nodes_arr.append(n)
        seq_nodes_arr.sort(key=lambda x: x.id)
        for idx in range(len(seq_nodes_arr)):
            if idx != len(seq_nodes_arr) - 1 and seq_nodes_arr[idx].tail == True:
                if seq_nodes_arr[idx + 1].head == True:
                    res.add_edge(seq_nodes_arr[idx], seq_nodes_arr[idx + 1])
                    res.add_edge(seq_nodes_arr[idx + 1], seq_nodes_arr[idx])
        return res

    def get_vocab(self, g):
        a = set()
        for n in list(g.nodes()):
            a.add(n.word)
        return a

    def get_adj(self, g):
        # reverse the direction
        adj_dict = {}
        for node, n_dict in g.adjacency():
            adj_dict[node.id] = []

        for node, n_dict in g.adjacency():
            for i in list(n_dict.items()):
                adj_dict[i[0].id].append(node.id)
        return adj_dict

    def get_constituency_graph(self, original_str):
        # whether to cut root
        parse_str = self.cut_root_node(self.parse(original_str))

        token_list = original_str.split()
        for punc in ['(', ')']:
            parse_str = parse_str.replace(punc, ' ' + punc + ' ')

        parse_list = (parse_str).split()
        
        res_graph = nx.DiGraph()
        pstack = Stack()
        idx = 0
        while idx < len(parse_list):
            if parse_list[idx] == '(':
                new_node = Node(word=parse_list[idx+1], id_=idx+1, type_=1)
                res_graph.add_node(new_node)
                pstack.push(new_node)

                if pstack.size() > 1:
                    node_2 = pstack.pop()
                    node_1 = pstack.pop()
                    res_graph.add_edge(node_1, node_2)
                    pstack.push(node_1)
                    pstack.push(node_2)
            elif parse_list[idx] == ')':
                pstack.pop()
            elif parse_list[idx] in token_list:
                new_node = Node(word=parse_list[idx], id_=idx, type_=0)
                node_1 = pstack.pop()
                if node_1.id != new_node.id:
                    res_graph.add_edge(node_1, new_node)
                pstack.push(node_1)
            idx += 1

        max_id = 0
        for n in res_graph.nodes():
            if n.type == 0 and n.id > max_id:
                max_id = n.id

        min_id = 99999
        for n in res_graph.nodes():
            if n.type == 0 and n.id < min_id:
                min_id = n.id

        for n in res_graph.nodes():
            if n.type == 0 and n.id == max_id:
                n.tail = True
            if n.type == 0 and n.id == min_id:
                n.head = True
        return res_graph

    def generate_batch_graph(self, output_file, string_batch):
        '''generate constituency graph'''
        graph_list = []
        max_node_size = 0
        for s in string_batch:
            # generate multiple graph or not
            if self.args.generate_mutiple_graph == 1:
                # print s
                s_arr = self.split_str(s)
                # print s.split()

                s_arr = []
                arr1 = s.split('\xef\xbc\x8e')
                for i in arr1:
                    s_arr.append(i.strip())

                g = self.cut_line_node(self.get_constituency_graph(s_arr[0].strip()))
                for sub_s in s_arr:
                    if sub_s != s_arr[0] and len(sub_s.strip()) > 0:
                        tmp = self.cut_line_node(self.get_constituency_graph(sub_s.strip()))
                        g = self.graph_connect(g, tmp)
            else:
                '''decide how to cut nodes'''
                g = self.cut_pos_node(self.get_constituency_graph(s))
                # g = self.cut_line_node(self.get_constituency_graph(s))
                # g = (get_constituency_graph(processor.featureExtract(s)))
            if len(list(g.nodes())) > max_node_size:
                max_node_size = len(list(g.nodes()))
            graph_list.append(g)

        info_list = []
        batch_size = len(string_batch)
        for index in range(batch_size):
            word_list = self.get_all_text(graph_list[index])
            word_len = len(self.get_seq_nodes(graph_list[index]))
            id_arr = self.get_all_id(graph_list[index])
            adj_dic = self.get_adj(graph_list[index])
            new_dic = {}

            # print id_arr
            # print adj_dic
            # print word_list

            # transform id to position in wordlist
            for k in list(adj_dic.keys()):
                new_dic[id_arr.index(k)] = [id_arr.index(x)
                                            for x in adj_dic[k]]

            # print new_dic

            info = {}

            g_ids = {}
            g_ids_features = {}
            g_adj = {}

            for idx in range(max_node_size):
                g_ids[idx] = idx
                if idx < len(word_list):
                    g_ids_features[idx] = word_list[idx]

                    # whether to link word nodes

                    # if idx <= word_len - 1:
                    #     if idx == 0:
                    #         new_dic[idx].append(idx + 1)
                    #     elif idx == word_len - 1:
                    #         new_dic[idx].append(idx - 1)
                    #     else:
                    #         new_dic[idx].append(idx - 1)
                    #         new_dic[idx].append(idx + 1)

                    g_adj[idx] = new_dic[idx]
                else:
                    g_ids_features[idx] = '<P>'
                    g_adj[idx] = []

            info['g_ids'] = g_ids
            info['g_ids_features'] = g_ids_features
            info['g_adj'] = g_adj
            # info['word_list'] = word_list
            # info['word_len'] = word_len

            info_list.append(info)

        with open(output_file, "a+") as f:
            # json.dump(info_list, f, indent=4)
            for idx in range(len(info_list)):
                f.write(json.dumps(info_list[idx]) + '\n')

        batch_vocab = []
        for x in graph_list:
            non_arr = self.nodes_to_string(self.get_non_seq_nodes(x)).split()
            for w in non_arr:
                if w not in batch_vocab:
                    batch_vocab.append(w)
        return batch_vocab

    def train_data_preprocess(self):
        time_start = time.time()
        word_manager = SymbolsManager(True)
        word_manager.init_from_file(
            "{}/vocab.q.txt".format(self.source_data_dir), self.min_freq, self.max_vocab_size)
        form_manager = SymbolsManager(True)
        form_manager.init_from_file(
            "{}/vocab.f.txt".format(self.source_data_dir), 0, self.max_vocab_size)
        print((word_manager.vocab_size))
        print((form_manager.vocab_size))

        data = []
        with open("{}/{}.txt".format(self.source_data_dir, "train"), "r") as f:
            for line in f:
                l_list = line.split("\t")
                w_list = l_list[0].strip().split(' ')
                r_list = form_manager.get_symbol_idx_for_list(
                    l_list[1].strip().split(' '))
                cur_tree = convert_to_tree(
                    r_list, 0, len(r_list), form_manager)

                data.append((w_list, r_list, cur_tree))
                # print [form_manager.get_idx_symbol(x) for x in cur_tree.to_list(form_manager)]

        out_graphfile = "{}/graph.train".format(self.output_data_dir)
        if os.path.exists(out_graphfile):
            os.remove(out_graphfile)
        # generate batch graph here
        if len(data) % self.batch_size != 0:
            n = len(data)
            for i in range(self.batch_size - len(data) % self.batch_size):
                data.insert(n-i-1, copy.deepcopy(data[n-i-1]))

        index = 0
        while index + self.batch_size <= len(data):
            # generate graphs with order and dependency information
            input_batch = [" ".join(data[index + idx][0])
                           for idx in range(self.batch_size)]
            new_vocab = self.generate_batch_graph(
                output_file=out_graphfile, string_batch=input_batch)
            for w in new_vocab:
                if w not in word_manager.symbol2idx:
                    word_manager.add_symbol(w)
                    print("{} Added.".format(w))
            index += self.batch_size
            print(index)
    #         if index >= 120:
    #             break

        out_datafile = "{}/train.pkl".format(self.output_data_dir)
        with open(out_datafile, "wb") as out_data:
            pkl.dump(data, out_data)

        out_mapfile = "{}/map.pkl".format(self.output_data_dir)
        with open(out_mapfile, "wb") as out_map:
            pkl.dump([word_manager, form_manager], out_map)

        print((word_manager.vocab_size))
        print((form_manager.vocab_size))

        time_end = time.time()
        print("time used:" + str(time_end - time_start))

    def test_data_preprocess(self):
        data = []
        managers = pkl.load(
            open("{}/map.pkl".format(self.output_data_dir), "rb"))
        _, form_manager = managers
        with open("{}/{}.txt".format(self.source_data_dir, "test"), "r") as f:
            for line in f:
                l_list = line.split("\t")
                w_list = l_list[0].strip().split(' ')
                r_list = form_manager.get_symbol_idx_for_list(
                    l_list[1].strip().split(' '))
                cur_tree = convert_to_tree(
                    r_list, 0, len(r_list), form_manager)
                data.append((w_list, r_list, cur_tree))

        out_datafile = "{}/test.pkl".format(self.output_data_dir)
        with open(out_datafile, "wb") as out_data:
            pkl.dump(data, out_data)

        out_graphfile = "{}/graph.test".format(self.output_data_dir)
        if os.path.exists(out_graphfile):
            os.remove(out_graphfile)

        index = 0
        while index + self.batch_size <= len(data):
            # generate graphs with order and dependency information
            input_batch = [" ".join(data[index + idx][0])
                           for idx in range(self.batch_size)]
            _ = self.generate_batch_graph(
                output_file=out_graphfile, string_batch=input_batch)
            index += self.batch_size

        if index != len(data):
            input_batch = [" ".join(data[idx][0])
                           for idx in range(index, len(data))]
            _ = self.generate_batch_graph(
                output_file=out_graphfile, string_batch=input_batch)

    def valid_data_preprocess(self):
        data = []
        managers = pkl.load(
            open("{}/map.pkl".format(self.output_data_dir), "rb"))
        _, form_manager = managers
        with open("{}/{}.txt".format(self.source_data_dir, "valid"), "r") as f:
            for line in f:
                l_list = line.split("\t")
                w_list = l_list[0].strip().split(' ')
                r_list = form_manager.get_symbol_idx_for_list(
                    l_list[1].strip().split(' '))
                cur_tree = convert_to_tree(
                    r_list, 0, len(r_list), form_manager)
                data.append((w_list, r_list, cur_tree))

        out_datafile = "{}/valid.pkl".format(self.output_data_dir)
        with open(out_datafile, "wb") as out_data:
            pkl.dump(data, out_data)

        out_graphfile = "{}/graph.valid".format(self.output_data_dir)
        if os.path.exists(out_graphfile):
            os.remove(out_graphfile)

        index = 0
        while index + self.batch_size <= len(data):
            # generate graphs with order and dependency information
            input_batch = [" ".join(data[index + idx][0])
                           for idx in range(self.batch_size)]
            _ = self.generate_batch_graph(
                output_file=out_graphfile, string_batch=input_batch)
            index += self.batch_size

        if index != len(data):
            input_batch = [" ".join(data[idx][0])
                           for idx in range(index, len(data))]
            _ = self.generate_batch_graph(
                output_file=out_graphfile, string_batch=input_batch)
        
    def GraphGenerateBegin(self):
        self.train_data_preprocess()
        # self.valid_data_preprocess()
        self.test_data_preprocess()
