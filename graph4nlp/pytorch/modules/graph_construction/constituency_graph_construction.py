import copy
import json
import random

import networkx as nx
import networkx.algorithms as nxalg
import numpy as np
import torch
import tqdm
from pythonds.basic.stack import Stack
from stanfordcorenlp import StanfordCoreNLP

from .base import StaticGraphConstructionBase
from .embedding_construction import EmbeddingConstruction
from ...data.data import GraphData

"""TODO: some design choice:
            - replace constituent tag "." with "const_period"
            - unify whether use lower case in vocab and parsing
"""


class ConstituencyBasedGraphConstruction(StaticGraphConstructionBase):
    """
    Class for constituency graph construction.

    ...

    Attributes
    ----------
    embedding_styles : (dict)
        #TODO: complete with new attributes with "yuchen". Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.

    vocab: (set, optional)
        Vocabulary including all words appeared in graphs.

    Methods
    -------

    topology(paragraph, nlp_processor, merge_strategy=None, edge_strategy=None)
        Generate graph structure with nlp parser like ``CoreNLP`` etc.

    construct_static_graph(parsed_object, sub_sentence_id, edge_strategy=None)
        Construct a single static graph from a single sentence, to be called by ``topology`` function.

    graph_connect(nx_graph_list, merge_strategy=None)
        Construct a merged graph from a list of graphs, to be called by ``topology`` function.

    embedding(node_attributes, edge_attributes)
        Generate node/edge embeddings from node/edge attributes through an embedding layer.

    forward(raw_text_data, nlp_parser)
        Generate graph topology and embeddings.
    """

    def __init__(self, embedding_style, hidden_emb_size, word_emb_size, vocab=None):
        super(ConstituencyBasedGraphConstruction,
              self).__init__(vocab, embedding_style, hidden_emb_size)
        self.vocab = vocab
        self.embedding_style = embedding_style
        self.constituency_embedding_layer = None
        self.word_emb_size = word_emb_size
        self.hidden_size = hidden_emb_size

    def topology(self,
                 paragraph,
                 nlp_processor,
                 merge_strategy=None,
                 edge_strategy=None):
        """topology This function generate a graph strcuture from a raw paragraph.

        Parameters
        ----------
        paragraph : string
            A string to be used to construct a static graph, can be composed of multiple strings

        nlp_processor : object
            A parser used to parse sentence string to parsing trees like dependency parsing tree or constituency parsing tree

        merge_strategy : str, optional
            Ways assigned for graph merge strategy: how to merge sentence-level graphs to a paragraph-level graph
            1) In a tail head way. 2) In a sequential way, by default None

        edge_strategy : str, optional
            Ways assigned for edge strategy: (1) 1/0: assign the weight (2) heterogeneous graph: type (3) as a node: process as node, by default None

        Returns
        -------
        GraphData
            A customized graph data structure
        """
        output_graph_list = []
        output = nlp_processor.annotate(
            paragraph.strip(),
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
        ret_graph = self.graph_connect(output_graph_list)
        return ret_graph

    def construct_static_graph(self,
                               parsed_object,
                               sub_sentence_id,
                               edge_strategy=None):
        parsed_sentence_data = parsed_object['parse']
        for punc in [u'(', u')']:
            parsed_sentence_data = parsed_sentence_data.replace(
                punc, ' ' + punc + ' ')
        parse_list = (parsed_sentence_data).split()

        res_graph = GraphData()
        pstack = Stack()
        idx = 0
        while idx < len(parse_list):
            if parse_list[idx] == '(':
                res_graph.add_nodes(1)
                self.vocab._add_words([parse_list[idx + 1]])
                res_graph.node_attributes[res_graph.get_node_num(
                ) - 1] = {'word': parse_list[idx + 1], 'wordid': self.vocab.word2index[parse_list[idx + 1]], 'type': 1, 'position': idx + 1, 'sentence': sub_sentence_id, 'tail': False, 'head': False}
                pstack.push(res_graph.get_node_num() - 1)
                if pstack.size() > 1:
                    node_2 = pstack.pop()
                    node_1 = pstack.pop()
                    res_graph.add_edge(node_1, node_2)
                    pstack.push(node_1)
                    pstack.push(node_2)
            elif parse_list[idx] == ')':
                pstack.pop()
            elif parse_list[idx + 1] == u')' and parse_list[idx] != u')':
                res_graph.add_nodes(1)
                self.vocab._add_words([parse_list[idx]])
                res_graph.node_attributes[res_graph.get_node_num(
                ) - 1] = {'word': parse_list[idx], 'wordid': self.vocab.word2index[parse_list[idx]], 'type': 0, 'position': idx, 'sentence': sub_sentence_id, 'tail': False, 'head': False}
                node_1 = pstack.pop()
                if node_1 != res_graph.get_node_num() - 1:
                    res_graph.add_edge(node_1, res_graph.get_node_num() - 1)
                pstack.push(node_1)
            idx += 1

        max_pos = 0
        for n in res_graph.node_attributes.values():
            if n['type'] == 0 and n['position'] > max_pos:
                max_pos = n['position']

        min_pos = 99999
        for n in res_graph.node_attributes.values():
            if n['type'] == 0 and n['position'] < min_pos:
                min_pos = n['position']

        for n in res_graph.node_attributes.values():
            if n['type'] == 0 and n['position'] == max_pos:
                n['tail'] = True
            if n['type'] == 0 and n['position'] == min_pos:
                n['head'] = True
        # print(res_graph.node_attributes)
        return res_graph

    def graph_connect(self, graph_list, merge_strategy=None):
        """
        Parameters
        ----------
        graph_list : list
            A graph list to be merged

        Returns
        -------
        GraphData
            A customized graph data structure
        """
        # _len_graph_ = len(graph_list)

        # output_graph = OurGraphData.union_all(graph_list)

        # for index in range(_len_graph_ - 1):
        #     head_node = OurGraphData.get_head_node(output_graph, index + 1)
        #     tail_node = OurGraphData.get_tail_node(output_graph, index)
        #     output_graph.add_edge(tail_node, head_node)
        # return output_graph
        return graph_list[0]

    def embedding(self, node_attributes, edge_attributes):
        node_emb, edge_emb = self.embedding_layer(
            node_attributes, edge_attributes)
        return node_emb, edge_emb

    def forward(self, raw_sentence_data, nlp_parser):
        output_graph = self.topology(raw_sentence_data,
                                     nlp_processor=nlp_parser)
        self.vocab.randomize_embeddings(self.word_emb_size)
        self.embedding_layer = EmbeddingConstruction(self.vocab, self.embedding_style['word_emb_type'], self.embedding_style[
                                                     'node_edge_level_emb_type'], self.embedding_style['graph_level_emb_type'], self.hidden_size)

        input_node_attributes = torch.LongTensor(
            [[i[1]['wordid'] for i in output_graph.node_attributes.items()]])
        graph_num = torch.LongTensor([1])
        node_num = torch.LongTensor([output_graph.get_node_num()])

        node_feat = self.embedding_layer(
            input_tensor=input_node_attributes, node_size=node_num, graph_size=graph_num)
        output_graph.node_features['node_feat'] = node_feat
        output_graph.node_features['node_meb'] = self.vocab.embeddings
        return output_graph
