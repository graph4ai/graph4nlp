import copy
import json

import torch
from pythonds.basic.stack import Stack
from stanfordcorenlp import StanfordCoreNLP

from graph4nlp.pytorch.data.data import GraphData, to_batch
from .base import StaticGraphConstructionBase
from .utils import CORENLP_TIMEOUT_SIGNATURE

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
        Specify embedding styles including ``single_token_item``, ``emb_strategy``, ``num_rnn_layers``, ``bert_model_name`` and ``bert_lower_case``.

    vocab: (set, optional)
        Vocabulary including all words appeared in graphs.

    Methods
    -------

    topology(raw_text_data, nlp_processor, merge_strategy=None, edge_strategy=None)
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

    def __init__(self, embedding_style, vocab, hidden_size, fix_word_emb=True, fix_bert_emb=True, word_dropout=None,
                 rnn_dropout=None, device=None):
        super(ConstituencyBasedGraphConstruction, self).__init__(word_vocab=vocab,
                                                                 embedding_styles=embedding_style,
                                                                 hidden_size=hidden_size,
                                                                 fix_word_emb=fix_word_emb,
                                                                 fix_bert_emb=fix_bert_emb,
                                                                 word_dropout=word_dropout,
                                                                 rnn_dropout=rnn_dropout,
                                                                 device=device)
        self.vocab = vocab
        assert (self.embedding_layer.device == device)
        self.device = self.embedding_layer.device

    @classmethod
    def parsing(cls, raw_text_data, nlp_processor, processor_args):
        '''
        Parameters
        ----------
        raw_text_data: str
        nlp_processor: StanfordCoreNLP
        split_hyphenated: bool
        normalize: bool
        '''
        output = nlp_processor.annotate(raw_text_data.strip(), properties=processor_args)
        if CORENLP_TIMEOUT_SIGNATURE in output:
            raise TimeoutError('CoreNLP timed out at input: \n{}\n This item will be skipped. '
                               'Please check the input or change the timeout threshold.'.format(raw_text_data))
        parsed_output = json.loads(output)['sentences']
        return parsed_output

    @classmethod
    def topology(cls,
                 raw_text_data,
                 nlp_processor,
                 processor_args,
                 merge_strategy=None,
                 edge_strategy=None,
                 verbase=True):
        """topology This function generate a graph strcuture from a raw text data.

        Parameters
        ----------
        raw_text_data : string
            A string to be used to construct a static graph, can be composed of multiple strings

        nlp_processor : object
            A parser used to parse sentence string to parsing trees like dependency parsing tree or constituency parsing tree

        merge_strategy : None or str, option=[None, "tailhead", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``: It will be the default option. We will do as ``"tailhead"``.
            ``"tailhead"``: Link the sub-graph  ``i``'s tail node with ``i+1``'s head node
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
        parsed_output = cls.parsing(raw_text_data, nlp_processor, processor_args)
        for index in range(len(parsed_output)):
            output_graph_list.append(
                cls._construct_static_graph(parsed_output[index], index))
        ret_graph = cls._graph_connect(output_graph_list)
        if verbase:
            print('--------------------------------------')
            for _edge in ret_graph.get_all_edges():
                print(ret_graph.nodes[_edge[0]].attributes['token'],
                      "\t---\t", ret_graph.nodes[_edge[1]].attributes['token'])
            print('--------------------------------------')

        return ret_graph

    @classmethod
    def _construct_static_graph(cls,
                                parsed_object,
                                sub_sentence_id,
                                edge_strategy=None,
                                sequential_link=True,
                                bisequential_link=True,
                                top_down=False,
                                add_pos_node=False):
        """construct single syntactic graph

        Parameters
        ----------
        parsed_object :
            A parsed object of single sentence after external parser like ``CoreNLP``

        sub_sentence_id : int
            To specify which sentence the ``parsed_object`` is in the ``raw_text_data``

        bisequential_link : bool
            Add bi-directional edges between word nodes, do not add if ``False``

        top_down : bool
            Edge direction between nodes in constituency tree, if ``True``, add edges from top to down

        add_pos_node : bool
            Add part-of-speech nodes or not
        """
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
                ) - 1] = {'token': parse_list[idx + 1], 'type': 1, 'position_id': None, 'sentence_id': sub_sentence_id,
                          'tail': False, 'head': False}
                pstack.push(res_graph.get_node_num() - 1)
                if pstack.size() > 1:
                    node_2 = pstack.pop()
                    node_1 = pstack.pop()
                    if top_down:
                        res_graph.add_edge(node_1, node_2)
                    else:
                        res_graph.add_edge(node_2, node_1)
                    pstack.push(node_1)
                    pstack.push(node_2)
            elif parse_list[idx] == ')':
                pstack.pop()
            elif parse_list[idx + 1] == u')' and parse_list[idx] != u')':
                cnt_word_node += 1
                if add_pos_node:
                    res_graph.add_nodes(1)
                res_graph.node_attributes[res_graph.get_node_num(
                ) - 1] = {'token': parse_list[idx], 'type': 0, 'position_id': cnt_word_node,
                          'sentence_id': sub_sentence_id, 'tail': False, 'head': False}
                node_1 = pstack.pop()
                if node_1 != res_graph.get_node_num() - 1:
                    if top_down:
                        res_graph.add_edge(node_1, res_graph.get_node_num() - 1)
                    else:
                        res_graph.add_edge(res_graph.get_node_num() - 1, node_1)
                pstack.push(node_1)
            idx += 1

        if sequential_link:
            _len_single_graph = res_graph.get_node_num()
            _cnt_node = 0
            while (True):
                node_1_idx = -1
                node_2_idx = -1
                for idx, node_attr in res_graph.node_attributes.items():
                    if node_attr['position_id'] == _cnt_node:
                        node_1_idx = idx
                    elif node_attr['position_id'] == _cnt_node + 1:
                        node_2_idx = idx
                if node_1_idx != -1 and node_2_idx != -1:
                    res_graph.add_edge(node_1_idx, node_2_idx)
                    if bisequential_link:
                        res_graph.add_edge(node_2_idx, node_1_idx)
                if _cnt_node >= _len_single_graph:
                    break
                _cnt_node += 1

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
        # print(res_graph.edges)
        return res_graph

    @classmethod
    def _graph_connect(cls, graph_list, merge_strategy=None, bisequential_link=True, reformalize=True):
        """
        Parameters
        ----------
        graph_list : list
            A graph list to be merged

        bisequential_link : bool
            whether add bi-direnctional links between word nodes

        reformalize : bool
            If true, separate word nodes and non-terminal nodes in ``graph.node_attributes`` and put word nodes in the front position

        bisequential_link : bool
            whether add bi-direnctional links between word nodes

        reformalize : bool
            If true, separate word nodes and non-terminal nodes in ``graph.node_attributes`` and put word nodes in the front position

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
                graph_i_nodes_attributes[dict_item[0] +
                                         len_merged_graph] = dict_item[1]
            merged_graph.node_attributes.update(graph_i_nodes_attributes)

        _cnt_edge_num = 0
        for g_idx in range(_len_graph_):
            tmp_edges = graph_list[g_idx].get_all_edges()
            current_node_num = graph_list[g_idx].get_node_num()
            for _edge in tmp_edges:
                merged_graph.add_edge(
                    _edge[0] + _cnt_edge_num, _edge[1] + _cnt_edge_num)
            _cnt_edge_num += current_node_num

        for index in range(_len_graph_ - 1):
            # get head node
            head_node = -1
            for node_attr_item in merged_graph.node_attributes.items():
                if node_attr_item[1]['sentence_id'] == index + 1 and node_attr_item[1]['type'] == 0 and \
                        node_attr_item[1]['head'] == True:
                    head_node = node_attr_item[0]
            # get tail node
            tail_node = -1
            for node_attr_item in merged_graph.node_attributes.items():
                if node_attr_item[1]['sentence_id'] == index and node_attr_item[1]['type'] == 0 and node_attr_item[1][
                    'tail'] == True:
                    tail_node = node_attr_item[0]
            merged_graph.add_edge(tail_node, head_node)
            if bisequential_link:
                merged_graph.add_edge(head_node, tail_node)
        if reformalize:
            new_dict_for_word_nodes = copy.deepcopy(merged_graph.node_attributes)

            dict_for_word_nodes = {}
            dict_for_non_ternimal_nodes = {}
            for i in new_dict_for_word_nodes.items():
                if i[1]['type'] == 0:
                    dict_for_word_nodes[i[0]] = i[1]
                else:
                    dict_for_non_ternimal_nodes[i[0]] = i[1]
            merged_graph.node_attributes.clear()
            merged_graph.node_attributes.update({**dict_for_word_nodes, **dict_for_non_ternimal_nodes})

            node_id_map = {}
            cnt = 0
            for i in merged_graph.node_attributes.items():
                node_id_map[i[0]] = cnt
                cnt += 1

            for i in range(merged_graph.get_edge_num()):
                merged_graph._edge_indices.src[i] = node_id_map[merged_graph._edge_indices.src[i]]
                merged_graph._edge_indices.tgt[i] = node_id_map[merged_graph._edge_indices.tgt[i]]
            reformalize_graph_attributes = {}
            for i in merged_graph.node_attributes.items():
                reformalize_graph_attributes[node_id_map[i[0]]] = copy.deepcopy(i[1])

            merged_graph.node_attributes.clear()
            merged_graph.node_attributes.update(reformalize_graph_attributes)
        return merged_graph

    def forward(self, batch_graphdata: list):
        node_size = []
        num_nodes = []
        num_word_nodes = []  # number of nodes that are extracted from the raw text in each graph

        for g in batch_graphdata:
            g.node_features['token_id'] = g.node_features['token_id'].to(self.device)
            num_nodes.append(g.get_node_num())
            num_word_nodes.append(len([1 for i in range(len(g.node_attributes)) if g.node_attributes[i]['type'] == 0]))
            node_size.extend([1 for i in range(num_nodes[-1])])

        batch_gd = to_batch(batch_graphdata)
        node_size = torch.Tensor(node_size).to(self.device).int()
        num_nodes = torch.Tensor(num_nodes).to(self.device).int()
        num_word_nodes = torch.Tensor(num_word_nodes).to(self.device).int()
        node_emb = self.embedding_layer(batch_gd, node_size, num_nodes, num_word_items=num_word_nodes)
        batch_gd.node_features["node_feat"] = node_emb
        return batch_gd

    def embedding(self, node_attributes, edge_attributes):
        node_emb, edge_emb = self.embedding_layer(
            node_attributes, edge_attributes)
        return node_emb, edge_emb
