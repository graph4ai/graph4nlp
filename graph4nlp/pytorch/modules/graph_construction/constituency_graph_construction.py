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
from .utility_functions import Node, UtilityFunctionsForGraph


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
    add_vocab(g)
        Expand vocabulary from graphs.
    
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
    def __init__(self, embedding_style, vocab=None):
        super(ConstituencyBasedGraphConstruction,
              self).__init__(embedding_style)
        self.vocab = vocab if vocab != None else set()
        self.word2idx = {}
        self.idx2word = []

    def add_vocab(self, g):
        vocab_list = UtilityFunctionsForGraph.get_all_text(g)
        for v in vocab_list:
            if v in self.vocab:
                pass
            else:
                self.vocab.add(v)
                self.word2idx[v] = len(self.vocab)
                self.idx2word.append(v)

    def topology(self,
                 paragraph,
                 nlp_processor,
                 merge_strategy=None,
                 edge_strategy=None):
        """
        Attributes
        ----------
        paragraph : (string)
            A string to be used to construct a static graph, can be composed of multiple strings.
        
        nlp_processer: (object)
            A parser used to parse sentence string to parsing trees like dependency parsing tree or constituency parsing tree.

        merge_strategy : (str)
            Ways assigned for graph merge strategy: how to merge sentence-level graphs to a paragraph-level graph.
            1) In a tail head way. 2) In a sequential way.

        edge_strategy: (str, optional)
            Ways assigned for edge strategy: (1) 1/0: assign the weight (2) heterogeneous graph: type (3) as a node: process as node
        """
        output_graph_list = []
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
        ret_graph = self.graph_connect(output_graph_list)
        self.add_vocab(ret_graph)
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
        return res_graph

    def graph_connect(self, nx_graph_list, merge_strategy=None):
        _len_graph_ = len(nx_graph_list)
        output_graph = nx.union_all(nx_graph_list)

        for index in range(_len_graph_ - 1):
            head_node = UtilityFunctionsForGraph.get_head_node(output_graph, index + 1)
            tail_node = UtilityFunctionsForGraph.get_tail_node(output_graph, index)
            output_graph.add_edge(tail_node, head_node)
        return output_graph

    def embedding(self, node_attributes, edge_attributes):
        node_emb, edge_emb = self.embedding_layer(node_attributes, edge_attributes)
        return node_emb, edge_emb

    def forward(self, raw_sentence_data, nlp_parser):
        output_graph = self.topology(raw_sentence_data,
                                     nlp_processor=nlp_parser)
        # TODO : implement or complete process from output_graph to node/edge embeddings and to dgl_graph finally.
        return output_graph

# code used for tests.
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
    print("syntactic parser ready\n-------------------")

    constituency_graph_gonstructor = ConstituencyBasedGraphConstruction(
        embedding_style=embedding_styles)
    output_graph = constituency_graph_gonstructor.forward(raw_data, nlp_parser)

    UtilityFunctionsForGraph.print_nodes(output_graph)
    print("-----------------------\nvocab size")
    print(len(constituency_graph_gonstructor.vocab))
