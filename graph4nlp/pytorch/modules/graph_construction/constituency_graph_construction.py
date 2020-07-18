import json
import copy
import tqdm
import torch
import random

import numpy as np
import networkx as nx
import networkx.algorithms as nxalg

from pythonds.basic.stack import Stack
from stanfordcorenlp import StanfordCoreNLP
from .base import StaticGraphConstructionBase


class Node():
    def __init__(self, word_, type_, id_, sentence_):
        # word: this node's text
        self.word = word_

        # type: 0 for word nodes, 1 for constituency nodes
        self.type = type_

        # id: unique identifier for every node
        self.id = id_

        self.head = False

        self.tail = False

        self.sentence = sentence_

    def __str__(self):
        return self.word + "_type_" + str(self.type) + "_id_" + str(
            self.id) + "_head_" + str(self.head) + "_tail_" + str(
                self.tail) + "_sentence_" + str(self.sentence)


class ConstituencyBasedGraphConstruction(StaticGraphConstructionBase):
    """
    Class for constituency graph construction.

    ...

    Attributes
    ----------
    embedding_styles : dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.
    
    vocab: set
        Vocabulary including all words appeared in graphs.

    Methods
    -------
    forward(raw_text_data)
        Generate graph topology and embeddings.

    """
    def __init__(self, embedding_style, vocab=None):
        super(ConstituencyBasedGraphConstruction,
              self).__init__(embedding_style)
        self.vocab = vocab if vocab != None else set()
        self.word2idx = {}
        self.idx2word = {}

    def add_vocab(self, nx_graph):
        pass

    def topology(self,
                 paragraph,
                 nlp_processor,
                 merge_strategy=None,
                 edge_strategy=None):
        output_graph_list = []

        print("-----------------------\nSource:", paragraph,
              "\n-----------------------")
        output = nlp_processor.annotate(
            paragraph.lower().strip(),
            properties={
                'annotators': "tokenize,ssplit,pos,parse",
                "tokenize.options":
                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                "tokenize.whitespace": False,
                'ssplit.isOneSentence': False,
                'outputFormat': 'json'
            })
        parsed_output = json.loads(output)['sentences']
        for index in range(len(parsed_output)):
            output_graph_list.append(
                self.construct_static_graph(parsed_output[index], index))
        return self.graph_connect(output_graph_list)

    def construct_static_graph(self,
                               parsed_object,
                               sub_sentence_id,
                               edge_strategy=None):
        parsed_sentence_data = parsed_object['parse']
        for punc in [u'(', u')']:
            parsed_sentence_data = parsed_sentence_data.replace(
                punc, ' ' + punc + ' ')
        parse_list = (parsed_sentence_data).split()

        res_graph = nx.DiGraph()
        pstack = Stack()
        idx = 0
        while idx < len(parse_list):
            if parse_list[idx] == '(':
                new_node = Node(word_=parse_list[idx + 1],
                                id_=idx + 1,
                                type_=1,
                                sentence_=sub_sentence_id)
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
            elif parse_list[idx + 1] == u')' and parse_list[idx] != u')':
                new_node = Node(word_=parse_list[idx],
                                id_=idx,
                                type_=0,
                                sentence_=sub_sentence_id)
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

        # print("--------------------all nodes-----------------")
        # self.print_nodes(res_graph)
        # print('--------------------word nodes----------------')
        # word_node = self.get_seq_nodes(res_graph)
        # for n in word_node:
        #     print(n)

        return res_graph

    def graph_connect(self, nx_graph_list, merge_strategy=None):
        
        _len_graph_ = len(nx_graph_list)
        output_graph = nx.union_all(nx_graph_list)

        for index in range(_len_graph_-1):
            head_node = self.get_head_node(output_graph, index+1)
            tail_node = self.get_tail_node(output_graph, index)
            output_graph.add_edge(tail_node, head_node)
        self.print_edges(output_graph)
        return output_graph

    def embedding(self, node_feat, edge_feat):
        node_emb, edge_emb = self.embedding_layer(node_feat, edge_feat)
        return node_emb, edge_emb

    def forward(self, raw_sentence_data, nlp_parser):
        output_graph = self.topology(raw_sentence_data,
                                     nlp_processor=nlp_parser)

        # node_feat = get_node_feat(nx_graph)
        # edge_feat = get_edge_feat(nx_graph)
        # node_emb, edge_emb = self.embedding(node_feat, edge_feat)
        # dgl_graph = convert2dgl(nx_graph, node_feat, edge_feat)
        # return dgl_graph

    # utility functions
    def get_head_node(self, g, sentence_id):
        for n in g.nodes():
            if (n.head == True) and (n.sentence == sentence_id):
                return n

    def get_tail_node(self, g, sentence_id):
        for n in g.nodes():
            if (n.tail == True) and (n.sentence == sentence_id):
                return n

    def cut_root_node(self, g):
        pass

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
            print(e[0].word, e[1].word), (e[0].id, e[1].id)

    def print_nodes(self, g):
        nodes_arr = list(g.nodes())
        for n in nodes_arr:
            print(n)


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    raw_data = "James went to the corner-shop. He want to buy some (eggs), <milk> and bread for breakfast."
    embedding_styles = {
        'word_emb_type': 'glove',
        'node_edge_level_emb_type': 'mean',
        'graph_level_emb_type': 'identity'
    }

    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)
    print("syntactic parser ready")

    constituency_graph_gonstructor = ConstituencyBasedGraphConstruction(
        embedding_style=embedding_styles)
    constituency_graph_gonstructor.forward(raw_data, nlp_parser)
    # def get_vocab(self, g):
    #     a = set()
    #     for n in list(g.nodes()):
    #         a.add(n.word)
    #     return a
