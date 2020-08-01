import json

from stanfordcorenlp import StanfordCoreNLP

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from .base import StaticGraphConstructionBase


class DependencyBasedGraphConstruction(StaticGraphConstructionBase):
    """
        Dependency-parsing-tree based graph construction class

    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(self, embedding_style, vocab, hidden_size=300, fix_word_emb=True, dropout=None, use_cuda=True):
        super(DependencyBasedGraphConstruction, self).__init__(word_vocab=vocab,
                                                               embedding_styles=embedding_style,
                                                               hidden_size=hidden_size,
                                                               fix_word_emb=fix_word_emb,
                                                               dropout=dropout, use_cuda=use_cuda)
        self.vocab = vocab
        self.verbase = 1

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
    def topology(cls, raw_text_data, nlp_processor, merge_strategy, edge_strategy):
        """
            Graph building method.

        Parameters
        ----------
        raw_text_data: str
            Raw text data, it can be multi-sentences.
        nlp_processor: StanfordCoreNLP
            NLP parsing tools
        merge_strategy: None or str, option=[None, "tailhead", "sequential", "user_define"]
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
        joint_graph: GraphData
            The merged graph data-structure.
        """
        cls.verbase = 1
        props = {
            'annotators': 'depparse',
            "tokenize.options":
                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': False,
            'outputFormat': 'json'
        }
        dep_json = nlp_processor.annotate(raw_text_data.strip(), properties=props)
        dep_dict = json.loads(dep_json)
        parsed_results = []
        node_id = 0
        for s_id, s in enumerate(dep_dict["sentences"]):
            parsed_sent = []
            unique_hash = {}
            node_id = 0

            for dep in s["basicDependencies"]:
                if cls.verbase > 0:
                    print(dep)
                if unique_hash.get((dep['governor'], dep['governorGloss'])) is None:
                    unique_hash[(dep['governor'], dep['governorGloss'])] = node_id
                    node_id += 1
                if unique_hash.get((dep['dependent'], dep['dependentGloss'])) is None:
                    unique_hash[(dep['dependent'], dep['dependentGloss'])] = node_id
                    node_id += 1
                dep_info = {
                    "edge_type": dep['dep'],
                    'src': {
                        'token': dep['governorGloss'],
                        'position_id': dep['governor'] - 1 if dep['governorGloss'] != "ROOT" else None,
                        'id': unique_hash[(dep['governor'], dep['governorGloss'])],
                        "sentence_id": s_id
                    },
                    'tgt': {
                        'token': dep['dependentGloss'],
                        'position_id': dep['dependent'] - 1 if dep['dependentGloss'] != "ROOT" else None,
                        'id': unique_hash[(dep['dependent'], dep['dependentGloss'])],
                        "sentence_id": s_id
                    }
                }
                if cls.verbase > 0:
                    print(dep_info)
                parsed_sent.append(dep_info)
            if cls.verbase > 0:
                print(node_id)
                print(len(parsed_sent))
            parsed_results.append({
                "graph_content": parsed_sent,
                "node_num": node_id
            })

        sub_graphs = []
        for sent_id, parsed_sent in enumerate(parsed_results):
            graph = cls._construct_static_graph(parsed_sent, edge_strategy=None)
            sub_graphs.append(graph)
        joint_graph = cls._graph_connect(sub_graphs, merge_strategy)
        return joint_graph

    def embedding(self, node_attributes, edge_attributes):
        node_emb, edge_emb = self.embedding_layer(
            node_attributes, edge_attributes)
        return node_emb, edge_emb

    @classmethod
    def _construct_static_graph(cls, parsed_object, edge_strategy=None):
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
                         It is not implemented yet.

        Returns
        -------
        graph: GraphData
            graph structure for single sentence
        """
        ret_graph = GraphData()
        node_num = parsed_object["node_num"]
        ret_graph.add_nodes(node_num)
        for dep_info in parsed_object["graph_content"]:
            if edge_strategy is None or edge_strategy == "homogeneous":
                ret_graph.add_edge(dep_info["src"]['id'], dep_info['tgt']['id'])
            elif edge_strategy == "heterogeneous":
                raise NotImplementedError()
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
                ret_graph.add_edge(dep_info['src']['id'], node_idx)
                ret_graph.add_edge(node_idx, dep_info['tgt']['id'])
            else:
                raise NotImplementedError()

            if dep_info["src"]['token'] != "ROOT":
                ret_graph.node_attributes[dep_info["src"]['id']]['type'] = 0
                ret_graph.node_attributes[dep_info["tgt"]['id']]['type'] = 0
            else:
                ret_graph.node_attributes[dep_info["src"]['id']]['type'] = 2  # 2 for dependency parsing tree
                ret_graph.node_attributes[dep_info["tgt"]['id']]['type'] = 2

            ret_graph.node_attributes[dep_info["src"]['id']]['token'] = dep_info["src"]['token']
            ret_graph.node_attributes[dep_info["tgt"]['id']]['token'] = dep_info['tgt']['token']
            ret_graph.node_attributes[dep_info["src"]['id']]['position_id'] = dep_info["src"]['position_id']
            ret_graph.node_attributes[dep_info["tgt"]['id']]['position_id'] = dep_info['tgt']['position_id']
            ret_graph.node_attributes[dep_info["src"]['id']]['sentence_id'] = dep_info["src"]['sentence_id']
            ret_graph.node_attributes[dep_info["tgt"]['id']]['sentence_id'] = dep_info["src"]['sentence_id']
            ret_graph.node_attributes[dep_info["src"]['id']]['head'] = False
            ret_graph.node_attributes[dep_info["tgt"]['id']]['head'] = False
            ret_graph.node_attributes[dep_info["src"]['id']]['tail'] = False
            ret_graph.node_attributes[dep_info["tgt"]['id']]['tail'] = False

            if dep_info["src"]['id'] == 0:
                # head
                ret_graph.node_attributes[dep_info["src"]['id']]['head'] = True
                ret_graph.node_attributes[dep_info["src"]['id']]['tail'] = False

            if dep_info["tgt"]['id'] == node_num - 1:
                # tail
                ret_graph.node_attributes[dep_info["tgt"]['id']]["head"] = False
                ret_graph.node_attributes[dep_info["tgt"]['id']]["tail"] = True

            # TODO: add edge_attributes
        return ret_graph

    @classmethod
    def _graph_connect(cls, nx_graph_list, merge_strategy=None):
        """
            This method will merge the sub-graphs into one graph.

        Parameters
        ----------
        nx_graph_list: list[GraphData]
            The list of all sub-graphs.
        merge_strategy: None or str, option=[None, "tailhead", "sequential", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``: It will be the default option. We will do as ``"tailhead"``.
            ``"tailhead"``: Link the sub-graph  ``i``'s tail node with ``i+1``'s head node
            ``"sequential"``: If sub-graph has ``a1, a2, ..., an`` nodes, and sub-graph has ``b1, b2, ..., bm`` nodes.
                              We will link ``a1, a2``, ``a2, a3``, ..., ``an-1, an``, \
                              ``an, b1``, ``b1, b2``, ..., ``bm-1, bm``.
            ``"user_define"``: We will give this option to the user. User can override this method to define your merge
                               strategy.

        Returns
        -------
        joint_graph: GraphData
            The merged graph structure.
        """
        if len(nx_graph_list) == 1:
            return nx_graph_list[0]
        node_num_list = [s_g.get_node_num() for s_g in nx_graph_list]
        node_num = sum(node_num_list)
        g = GraphData()
        g.add_nodes(node_num)
        node_idx_off = 0

        # copy edges
        for s_g in nx_graph_list:
            for edge in s_g.get_all_edges():
                src, tgt = edge
                g.add_edge(src + node_idx_off, tgt + node_idx_off)
            tmp = {}
            for key, value in s_g.node_attributes.items():
                tmp[key + node_idx_off] = value
            g.node_attributes.update(tmp)
            node_idx_off += s_g.get_node_num()

        head_g = -1
        tail_g = -1

        headtail_list = []

        head = -1
        tail = -1

        node_idx_off = 0
        for i in range(len(nx_graph_list)):
            for node_idx, node_attrs in nx_graph_list[i].node_attributes.items():
                print(node_attrs, "-----")
                if node_attrs['head'] is True:
                    head = node_idx + node_idx_off
                elif node_attrs['tail'] is True:
                    tail = node_idx + node_idx_off
            assert head != -1
            assert tail != -1
            headtail_list.append((head, tail))
            head = -1
            tail = -1

        for n_node in node_num_list:
            headtail_list.append((node_idx_off, node_idx_off + n_node - 1))
            node_idx_off += n_node

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
        elif merge_strategy == "sequential":
            src_list = []
            tgt_list = []
            node_idx_off = 0
            for s_g_idx, n_node in enumerate(node_num_list):
                src_list.extend(list(range(node_idx_off, node_idx_off + n_node - 1)))
                tgt_list.extend(list(range(node_idx_off + 1, node_idx_off + n_node)))
                if s_g_idx != 0:
                    src_list.append(node_idx_off - 1)
                    tgt_list.append(node_idx_off)
                node_idx_off += n_node
            if cls.verbase > 0:
                print("merged edges")
                print("src list:", src_list)
                print("tgt list:", tgt_list)
            g.add_edges(src_list, tgt_list)
        else:
            # TODO: add two merge strategy
            raise NotImplementedError()


        for node_idx, node_attrs in g.node_attributes.items():
            node_attrs['head'] = node_idx == head_g
            node_attrs['tail'] = node_idx == tail_g


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
            print("-----------------------------")
            print("merged graph")
            print("node_num: {}".format(g.get_node_num()))
            for i in range(g.get_node_num()):
                print(g.get_node_attrs(i))
            print("edge_num: {}".format(g.get_edge_num()))
            print(g.get_all_edges())

        return g

    def forward(self, feat):
        pass
