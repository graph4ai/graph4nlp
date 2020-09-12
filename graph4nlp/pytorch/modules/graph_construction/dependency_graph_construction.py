import copy
import json

import torch
from stanfordcorenlp import StanfordCoreNLP

from .base import StaticGraphConstructionBase
from ...data.data import GraphData, to_batch


class DependencyBasedGraphConstruction(StaticGraphConstructionBase):
    """
        Dependency-parsing-tree based graph construction class

    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``single_token_item``, ``emb_strategy``, ``num_rnn_layers``, ``bert_model_name`` and ``bert_lower_case``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(self, embedding_style, vocab, hidden_size=300, fix_word_emb=True, fix_bert_emb=True, word_dropout=None, rnn_dropout=None, device=None):
        super(DependencyBasedGraphConstruction, self).__init__(word_vocab=vocab,
                                                               embedding_styles=embedding_style,
                                                               hidden_size=hidden_size,
                                                               fix_word_emb=fix_word_emb,
                                                               fix_bert_emb=fix_bert_emb,
                                                               word_dropout=word_dropout,
                                                               rnn_dropout=rnn_dropout,
                                                               device=device)
        self.vocab = vocab
        self.verbase = 1
        self.device = self.embedding_layer.device

    def add_vocab(self, g):
        """
            Add node tokens appeared in graph g to vocabulary.

        Parameters
        ----------
        g: GraphData
            Graph data-structure.

        """
        for i in range(g.get_node_num()):
            attr = g.get_node_attrs(i)[i]
            self.vocab.word_vocab._add_words([attr["token"]])

    @classmethod
    def parsing(cls, raw_text_data, nlp_processor, processor_args):
        '''

        Parameters
        ----------
        raw_text_data: str
        nlp_processor: StanfordCoreNLP
        processor_args: dict

        Returns
        -------
        parsed_results: list[dict]
            Each sentence is a dict. All sentences are packed by a list.
            key, value
            "node_num": int
                the node amount
            "node_content": list[dict]
                The list consisting node information. Each node is organized by a dict.
                'token': str
                    word token
                'position_id': int
                    the word's position id in original sentence. eg: I am a dog. position_id: 0, 1, 2, 3
                'id': int,
                    the node token's id which will be used in GraphData
                "sentence_id": int
                    The sentence's id in the whole text.
            "graph_content": list[dict]
                The list consisting edge information. Each edge is organized by a dict.
                "edge_type": str
                    The edge type token, eg: 'nsubj'
                'src': int
                    The source node ``id``
                'tgt': int
                    The target node ``id``
        '''
        dep_json = nlp_processor.annotate(raw_text_data.strip(), properties=processor_args)
        dep_dict = json.loads(dep_json)

        parsed_results = []
        node_id = 0
        for s_id in range(len(dep_dict["sentences"])):
            parsed_sent = []
            node_item = []
            unique_hash = {}
            node_id = 0

            for tokens in dep_dict["sentences"][s_id]['tokens']:
                unique_hash[(tokens['index'], tokens['word'])] = node_id
                node = {
                    'token': tokens["word"],
                    'position_id': tokens["index"] - 1,
                    'id': node_id,
                    "sentence_id": s_id
                }
                node_item.append(node)
                node_id += 1

            for dep in dep_dict["sentences"][s_id]["basicDependencies"]:
                if cls.verbase > 0:
                    print(dep)

                if dep['governorGloss'] == "ROOT":
                    continue

                if dep['dependentGloss'] == "ROOT":
                    continue

                dep_info = {
                    "edge_type": dep['dep'],
                    'src': unique_hash[(dep['governor'], dep['governorGloss'])],
                    'tgt': unique_hash[(dep['dependent'], dep['dependentGloss'])]
                }
                if cls.verbase > 0:
                    print(dep_info)
                parsed_sent.append(dep_info)
            if cls.verbase > 0:
                print(node_id)
                print(len(parsed_sent))
            parsed_results.append({
                "graph_content": parsed_sent,
                "node_content": node_item,
                "node_num": node_id
            })
            return parsed_results

    @classmethod
    def topology(cls, raw_text_data, nlp_processor, processor_args, merge_strategy, edge_strategy, sequential_link=True, verbase=0):
        """
            Graph building method.

        Parameters
        ----------
        raw_text_data: str or list[list]
            Raw text data, it can be multi-sentences.
            When it is ``str`` type, it is the raw text.
            When it is ``list[list]`` type, it is the tokenized token lists.
        nlp_processor: StanfordCoreNLP
            NLP parsing tools
        processor_args: dict
            The configure dict for StanfordCoreNLP.annotate
        merge_strategy: None or str, option=[None, "tailhead", "user_define"]
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
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node (``i``, ``k``) and (``k``, ``j``).

        sequential_link: bool, default=True
            Whether to link node tokens sequentially (note that it is bidirectional)
        verbase: int, default=0
            Whether to output log infors. Set 1 to output more infos.
        Returns
        -------
        joint_graph: GraphData
            The merged graph data-structure.
        """
        cls.verbase = verbase

        parsed_results = cls.parsing(raw_text_data=raw_text_data, nlp_processor=nlp_processor,
                                     processor_args=processor_args)

        sub_graphs = []
        for sent_id, parsed_sent in enumerate(parsed_results):
            graph = cls._construct_static_graph(parsed_sent, edge_strategy=edge_strategy,
                                                sequential_link=sequential_link)
            sub_graphs.append(graph)
        joint_graph = cls._graph_connect(sub_graphs, merge_strategy)
        return joint_graph

    @classmethod
    def _construct_static_graph(cls, parsed_object, edge_strategy=None, sequential_link=True):
        """
            Build dependency-parsing-tree based graph for single sentence.

        Parameters
        ----------
        parsed_object: dict
            The parsing tree.
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

        Returns
        -------
        graph: GraphData
            graph structure for single sentence
        """
        ret_graph = GraphData()
        node_num = parsed_object["node_num"]
        assert node_num > 0
        ret_graph.add_nodes(node_num)
        head_node = 0
        tail_node = node_num - 1

        # insert node attributes
        node_objects = parsed_object["node_content"]
        for node in node_objects:
            ret_graph.node_attributes[node['id']]['type'] = 0
            ret_graph.node_attributes[node['id']]['token'] = node['token']
            ret_graph.node_attributes[node['id']]['position_id'] = node['position_id']
            ret_graph.node_attributes[node['id']]['sentence_id'] = node['sentence_id']
            ret_graph.node_attributes[node['id']]['head'] = False
            ret_graph.node_attributes[node['id']]['tail'] = False

        for dep_info in parsed_object["graph_content"]:
            if edge_strategy is None or edge_strategy == "homogeneous":
                ret_graph.add_edge(dep_info["src"], dep_info['tgt'])
            elif edge_strategy == "heterogeneous":
                ret_graph.add_edge(dep_info["src"], dep_info['tgt'])
                edge_idx = ret_graph.edge_ids(dep_info["src"], dep_info['tgt'])[0]
                ret_graph.edge_attributes[edge_idx]["token"] = dep_info["edge_type"]
            elif edge_strategy == "as_node":
                # insert a node
                node_idx = ret_graph.get_node_num()
                ret_graph.add_nodes(1)
                ret_graph.node_attributes[node_idx]['type'] = 3  # 3 for edge node
                ret_graph.node_attributes[node_idx]['token'] = dep_info['edge_type']
                ret_graph.node_attributes[node_idx]['position_id'] = None
                ret_graph.node_attributes[node_idx]['head'] = False
                ret_graph.node_attributes[node_idx]['tail'] = False
                # add edge infos
                ret_graph.add_edge(dep_info['src'], node_idx)
                ret_graph.add_edge(node_idx, dep_info['tgt'])
            else:
                raise NotImplementedError()
        ret_graph.node_attributes[head_node]['head'] = True
        ret_graph.node_attributes[tail_node]['tail'] = True

        sequential_list = [i for i in range(node_num)]

        if sequential_list and len(sequential_list) > 1:
            for st, ed in zip(sequential_list[:-1], sequential_list[1:]):
                try:
                    ret_graph.edge_ids(st, ed)
                except:
                    ret_graph.add_edge(st, ed)
                try:
                    ret_graph.edge_ids(ed, st)
                except:
                    ret_graph.add_edge(ed, st)
        return ret_graph

    @classmethod
    def _graph_connect(cls, nx_graph_list, merge_strategy=None):
        """
            This method will merge the sub-graphs into one graph.

        Parameters
        ----------
        nx_graph_list: list[GraphData]
            The list of all sub-graphs.
        merge_strategy: None or str, option=[None, "tailhead", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``: It will be the default option. We will do as ``"tailhead"``.
            ``"tailhead"``: Link the sub-graph  ``i``'s tail node with ``i+1``'s head node
            ``"user_define"``: We will give this option to the user. User can override this method to define your merge
                               strategy.

        Returns
        -------
        joint_graph: GraphData
            The merged graph structure.
        """
        if cls.verbase > 0:
            print("sub_graph print")
            for i, s_g in enumerate(nx_graph_list):
                print("-------------------------")
                print("sub-graph: {}".format(i))
                print("node_num: {}".format(s_g.get_node_num()))
                for i in range(s_g.get_node_num()):
                    print(s_g.get_node_attrs(i))
                print("edge_num: {}".format(s_g.get_edge_num()))
                print(s_g.get_all_edges())
                for i in range(s_g.get_edge_num()):
                    print(i, s_g.edge_attributes[i])
            print("*************************************")
        if len(nx_graph_list) == 1:
            return nx_graph_list[0]
        elif len(nx_graph_list) == 0:
            raise RuntimeError("There is no graph needed to merge.")
        node_num_list = [s_g.get_node_num() for s_g in nx_graph_list]
        node_num = sum(node_num_list)
        g = GraphData()
        g.add_nodes(node_num)
        node_idx_off = 0

        # copy edges
        for s_g in nx_graph_list:
            for edge in s_g.get_all_edges():
                src, tgt = edge
                edge_idx_old = s_g.edge_ids(src, tgt)[0]
                g.add_edge(src + node_idx_off, tgt + node_idx_off)
                edge_idx_new = g.edge_ids(src + node_idx_off, tgt + node_idx_off)[0]
                if cls.verbase > 0:
                    print(edge_idx_new, edge_idx_old)
                    print(s_g.edge_attributes[edge_idx_old], "--------")
                g.edge_attributes[edge_idx_new] = copy.deepcopy(s_g.edge_attributes[edge_idx_old])
            tmp = {}
            for key, value in s_g.node_attributes.items():
                tmp[key + node_idx_off] = copy.deepcopy(value)
            g.node_attributes.update(tmp)
            node_idx_off += s_g.get_node_num()

        headtail_list = []

        head = -1
        tail = -1

        node_idx_off = 0
        for i in range(len(nx_graph_list)):
            for node_idx, node_attrs in nx_graph_list[i].node_attributes.items():
                if node_attrs['head'] is True:
                    head = node_idx + node_idx_off
                if node_attrs['tail'] is True:
                    tail = node_idx + node_idx_off
            assert head != -1
            assert tail != -1
            headtail_list.append((head, tail))
            head = -1
            tail = -1
            node_idx_off += node_num_list[i]

        head_g = headtail_list[0][0]
        tail_g = headtail_list[-1][1]

        if merge_strategy is None or merge_strategy == "tailhead":

            src_list = []
            tgt_list = []

            for i in range(len(headtail_list) - 1):
                src_list.append(headtail_list[i][1])
                tgt_list.append(headtail_list[i + 1][0])
            if cls.verbase > 0:
                print("merged edges")
                print("src list:", src_list)
                print("tgt list:", tgt_list)
            g.add_edges(src_list, tgt_list)
        else:
            raise NotImplementedError()

        for node_idx, node_attrs in g.node_attributes.items():
            node_attrs['head'] = node_idx == head_g
            node_attrs['tail'] = node_idx == tail_g

        if cls.verbase > 0:
            print("-----------------------------")
            print("merged graph")
            print("node_num: {}".format(g.get_node_num()))
            for i in range(g.get_node_num()):
                print(g.get_node_attrs(i))
            print("edge_num: {}".format(g.get_edge_num()))
            print(g.get_all_edges())
            for i in range(g.get_edge_num()):
                print(i, g.edge_attributes[i])
        return g

    def forward(self, batch_graphdata: list):
        node_size = []
        num_nodes = []
        num_word_nodes = [] # number of nodes that are extracted from the raw text in each graph

        for g in batch_graphdata:
            g.node_features['token_id'] = g.node_features['token_id'].to(self.device)
            num_nodes.append(g.get_node_num())
            num_word_nodes.append(len([1 for i in range(len(g.node_attributes)) if g.node_attributes[i]['type'] == 0]))
            node_size.extend([1 for i in range(num_nodes[-1])])

        batch_gd = to_batch(batch_graphdata)
        b_node = batch_gd.get_node_num()
        assert b_node == sum(num_nodes), print(b_node, sum(num_nodes))
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
