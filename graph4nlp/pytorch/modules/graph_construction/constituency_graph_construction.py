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
        Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.

    vocab: (set, optional)
        Vocabulary including all words appeared in graphs.

    Methods
    -------

    topology(paragraph, nlp_processor, merge_strategy=None, edge_strategy=None)
        Generate graph structure with nlp parser like ``CoreNLP`` etc.

    _construct_static_graph(parsed_object, sub_sentence_id, edge_strategy=None)
        Construct a single static graph from a single sentence, to be called by ``topology`` function.

    _graph_connect(nx_graph_list, merge_strategy=None)
        Construct a merged graph from a list of graphs, to be called by ``topology`` function.

    embedding(node_attributes, edge_attributes)
        Generate node/edge embeddings from node/edge attributes through an embedding layer.

    forward(raw_text_data, nlp_parser)
        Generate graph topology and embeddings.
    """

    def __init__(self, word_vocab, embedding_styles, hidden_size,
                 fix_word_emb=True, dropout=None, use_cuda=True):
        super(ConstituencyBasedGraphConstruction, self).__init__(word_vocab,
                                                                 embedding_styles,
                                                                 hidden_size,
                                                                 fix_word_emb=fix_word_emb,
                                                                 dropout=dropout,
                                                                 use_cuda=use_cuda)
    @classmethod
    def topology(cls,
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

        merge_strategy : None or str, option=[None, "tailhead", "sequential", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``: It will be the default option. We will do as ``"tailhead"``.
            ``"tailhead"``: Link the sub-graph  ``i``'s tail node with ``i+1``'s head node
            ``"sequential"``: If sub-graph has ``a1, a2, ..., an`` nodes, and sub-graph has ``b1, b2, ..., bm`` nodes.
                              We will link ``a1, a2``, ``a2, a3``, ..., ``an-1, an``, \
                              ``an, b1``, ``b1, b2``, ..., ``bm-1, bm``.
            ``"user_define"``: We will give this option to the user. User can override this method to define your merge
                               strategy.

        edge_strategy: None or str, option=[None, "homogeneous", "heterogeneous", "as_node"]
            Strategy to process edge.
            ``None``: It will be the default option. We will do as ``"homogeneous"``.
            ``"homogeneous"``: We will drop the edge type information.
                               If there is a linkage among node ``i`` and node ``j``, we will add an edge whose weight
                               is ``1.0``. Otherwise there is no edge.
            ``heterogeneous``: We will keep the edge type information.
                               An edge will have type information like ``n_subj``.
                               It is not implemented yet.
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node (``i``, ``k``) and (``k``, ``j``).
                         It is not implemented yet.

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
                cls._construct_static_graph(parsed_output[index], index))
        ret_graph = cls._graph_connect(output_graph_list)
        return ret_graph

    @classmethod
    def _construct_static_graph(cls,
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
        cnt_word_node = -1
        while idx < len(parse_list):
            if parse_list[idx] == '(':
                res_graph.add_nodes(1)
                res_graph.node_attributes[res_graph.get_node_num(
                ) - 1] = {'token': parse_list[idx + 1], 'type': 1, 'position_id': None, 'sentence_id': sub_sentence_id, 'tail': False, 'head': False}
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
                cnt_word_node += 1
                res_graph.add_nodes(1)
                res_graph.node_attributes[res_graph.get_node_num(
                ) - 1] = {'token': parse_list[idx], 'type': 0, 'position_id': cnt_word_node, 'sentence_id': sub_sentence_id, 'tail': False, 'head': False}
                node_1 = pstack.pop()
                if node_1 != res_graph.get_node_num() - 1:
                    res_graph.add_edge(node_1, res_graph.get_node_num() - 1)
                pstack.push(node_1)
            idx += 1

        max_pos = 0
        for n in res_graph.node_attributes.values():
            if n['type'] == 0 and n['position_id'] > max_pos:
                max_pos = n['position_id']

        min_pos = 99999
        for n in res_graph.node_attributes.values():
            if n['type'] == 0 and n['position_id'] < min_pos:
                min_pos = n['position_id']

        for n in res_graph.node_attributes.values():
            if n['type'] == 0 and n['position_id'] == max_pos:
                n['tail'] = True
            if n['type'] == 0 and n['position_id'] == min_pos:
                n['head'] = True
        # print(res_graph.node_attributes)
        return res_graph

    @classmethod
    def _graph_connect(cls, graph_list, merge_strategy=None):
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
        _len_graph_ = len(graph_list)
        merged_graph = GraphData()
        for index in range(_len_graph_):
            len_merged_graph = merged_graph.get_node_num()
            graph_i_nodes_attributes = {}
            for dict_item in graph_list[index].node_attributes.items():
                graph_i_nodes_attributes[dict_item[0] + len_merged_graph] = dict_item[1]
            merged_graph.node_attributes.update(graph_i_nodes_attributes)

        for index in range(_len_graph_ - 1):
            # get head node
            head_node = -1
            for node_attr_item in merged_graph.node_attributes.items():
                if node_attr_item[1]['sentence_id'] == index + 1 and node_attr_item[1]['type'] == 0 and node_attr_item[1]['head'] == True:
                    head_node = node_attr_item[0]
            # get tail node
            tail_node = -1
            for node_attr_item in merged_graph.node_attributes.items():
                if node_attr_item[1]['sentence_id'] == index and node_attr_item[1]['type'] == 0 and node_attr_item[1]['tail'] == True:
                    tail_node = node_attr_item[0]
            merged_graph.add_edge(tail_node, head_node)
        return merged_graph

    def embedding(self, node_attributes, edge_attributes):
        node_emb, edge_emb = self.embedding_layer(
            node_attributes, edge_attributes)
        return node_emb, edge_emb

    def forward(self, raw_sentence_data, nlp_parser):
        output_graph = self.topology(raw_sentence_data,
                                     nlp_processor=nlp_parser)
        # self.vocab.randomize_embeddings(self.word_emb_size)
        # self.embedding_layer = EmbeddingConstruction(self.vocab, self.embedding_style['word_emb_type'], self.embedding_style[
        #                                              'node_edge_level_emb_type'], self.embedding_style['graph_level_emb_type'], self.hidden_size)

        # input_node_attributes = torch.LongTensor(
        #     [[i[1]['wordid'] for i in output_graph.node_attributes.items()]])
        # graph_num = torch.LongTensor([1])
        # node_num = torch.LongTensor([output_graph.get_node_num()])

        # node_feat = self.embedding_layer(
        #     input_tensor=input_node_attributes, node_size=node_num, graph_size=graph_num)
        # output_graph.node_features['node_feat'] = node_feat
        # output_graph.node_features['node_meb'] = self.vocab.embeddings
        return output_graph
