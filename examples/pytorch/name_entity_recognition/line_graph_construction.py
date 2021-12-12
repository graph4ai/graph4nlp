import copy
import itertools
from collections import Counter

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.modules.graph_construction.base import StaticGraphConstructionBase


class LineBasedGraphConstruction(StaticGraphConstructionBase):
    """
        Line based graph construction class
    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type``
        and ``graph_level_emb_type``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(
        self,
        vocab,
    ):
        super(LineBasedGraphConstruction, self).__init__()
        self.vocab = vocab
        self.verbose = 1

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
    def parsing(cls, raw_text_data, tokenizer=None):
        """
        Parameters
        ----------
        raw_text_data: list of of word tokens or string of sequence
        tokenizer: the tokenizer will be used if raw_text_data is a str; if None, use the default \
            tokenizoer will be used. The output of the required tokenizer should be a list \
            of tokens.
        Returns
        -------
        parsed_results: list of dict
            key, value
            "node_num": int
                the node amount
            "node_content": list[dict]
                The list consisting node information. Each node is organized by a dict.
                'token': str
                    word token
                'position_id': int
                    the word's position id in original sentence. eg: I am a dog.
                    position_id: 0, 1, 2, 3
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
        """
        if isinstance(raw_text_data, str):
            if tokenizer is None:
                raw_text_data = raw_text_data.split(" ")
            else:
                raw_text_data = tokenizer(raw_text_data)
        parsed_results = []
        # for sent_id in range(len(raw_text_data)):
        parsed_sent = {}
        parsed_sent["graph_content"] = None
        parsed_sent["node_content"] = []
        sent = raw_text_data  # [sent_id]
        assert len(sent) >= 2
        parsed_sent["node_num"] = len(sent)
        # generate node content in a text
        for token_id in range(len(sent)):
            node = {}
            node["token"] = sent[token_id]
            node["sentence_id"] = 0
            node["id"] = token_id
            node["position_id"] = token_id
            parsed_sent["node_content"].append(node)
        parsed_results.append(parsed_sent)

        return parsed_results

    @classmethod
    def static_topology(
        cls,
        raw_text_data,
        nlp_processor,
        processor_args,
        merge_strategy,
        edge_strategy,
        split_hyphenated=False,
        normalize=False,
        sequential_link=True,
        verbose=0,
    ):
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
        merge_strategy: None or str, option=[None, "tailhead", "sequential", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``: It will be the default option. We will do as ``"tailhead"``.
            ``"tailhead"``: Link the sub-graph  ``i``'s tail node with ``i+1``'s head node
            ``"sequential"``: If sub-graph has ``a1, a2, ..., an`` nodes, and sub-graph has
            ``b1, b2, ..., bm`` nodes.
                              We will link ``a1, a2``, ``a2, a3``, ..., ``an-1, an``, \
                              ``an, b1``, ``b1, b2``, ..., ``bm-1, bm``.
            ``"user_define"``: We will give this option to the user. User can override this
            method to define your merge strategy.
        edge_strategy: None or str, option=[None, "homogeneous", "heterogeneous", "as_node"]
            Strategy to process edge.
            ``None``: It will be the default option. We will do as ``"homogeneous"``.
            ``"homogeneous"``: We will drop the edge type information.
                               If there is a linkage among node ``i`` and node ``j``, we will
                               add an edge whose weight is ``1.0``. Otherwise there is no edge.
            ``heterogeneous``: We will keep the edge type information.
                               An edge will have type information like ``n_subj``.
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node (``i``, ``k``)
                         and (``k``, ``j``).
        split_hyphenated: bool, default=False
            Whether or not to tokenize segments of hyphenated words separately (“school” “-“ “aged”,
            “frog” “-“ “lipped”)
        normalize: bool, default=False
            Whether to convert bracket (`(`) to  -LRB-, and etc.
        sequential_link: bool, default=True
            Whether to link node tokens sequentially (note that it is bidirectional)
        verbose: int, default=0
            Whether to output log infors. Set 1 to output more infos.
        Returns
        -------
        joint_graph: GraphData
            The merged graph data-structure.
        """
        cls.verbose = verbose

        parsed_results = cls.parsing(raw_text_data=raw_text_data)

        sub_graphs = []
        for _, parsed_sent in enumerate(parsed_results):
            graph = cls._construct_static_graph(
                parsed_sent, edge_strategy=edge_strategy, sequential_link=sequential_link
            )
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
                               If there is a linkage among node ``i`` and node ``j``, we will
                               add an edge whose weight is ``1.0``. Otherwise there is no edge.
            ``heterogeneous``: We will keep the edge type information.
                               An edge will have type information like ``n_subj``.
                               It is not implemented yet.
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node (``i``, ``k``)
                         and (``k``, ``j``).
                         It is not implemented yet.
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
            ret_graph.node_attributes[node["id"]]["type"] = 0
            ret_graph.node_attributes[node["id"]]["token"] = node["token"]
            ret_graph.node_attributes[node["id"]]["position_id"] = node["position_id"]
            ret_graph.node_attributes[node["id"]]["sentence_id"] = node["sentence_id"]
            ret_graph.node_attributes[node["id"]]["head"] = False
            ret_graph.node_attributes[node["id"]]["tail"] = False

        #        for dep_info in parsed_object["graph_content"]:
        #            if edge_strategy is None or edge_strategy == "homogeneous":
        #                ret_graph.add_edge(dep_info["src"], dep_info['tgt'])
        #            elif edge_strategy == "heterogeneous":
        #                ret_graph.add_edge(dep_info["src"], dep_info['tgt'])
        #                edge_idx = ret_graph.edge_ids(dep_info["src"], dep_info['tgt'])[0]
        #                ret_graph.edge_attributes[edge_idx]["token"] = dep_info["edge_type"]
        #            elif edge_strategy == "as_node":
        #                # insert a node
        #                node_idx = ret_graph.get_node_num()
        #                ret_graph.add_nodes(1)
        #                ret_graph.node_attributes[node_idx]['type'] = 3  # 3 for edge node
        #                ret_graph.node_attributes[node_idx]['token'] = dep_info['edge_type']
        #                ret_graph.node_attributes[node_idx]['position_id'] = None
        #                ret_graph.node_attributes[node_idx]['head'] = False
        #                ret_graph.node_attributes[node_idx]['tail'] = False
        #                # add edge infos
        #                ret_graph.add_edge(dep_info['src'], node_idx)
        #                ret_graph.add_edge(node_idx, dep_info['tgt'])
        #            else:
        #                raise NotImplementedError()
        ret_graph.node_attributes[head_node]["head"] = True
        ret_graph.node_attributes[tail_node]["tail"] = True

        sequential_list = list(range(node_num))

        if sequential_list and len(sequential_list) > 1:
            for st, ed in zip(sequential_list[:-1], sequential_list[1:]):
                try:
                    ret_graph.edge_ids(st, ed)
                except Exception:
                    ret_graph.add_edge(st, ed)
                try:
                    ret_graph.edge_ids(ed, st)
                except Exception:
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
        merge_strategy: None or str, option=[None, "tailhead", "sequential", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``: It will be the default option. We will do as ``"tailhead"``.
            ``"tailhead"``: Link the sub-graph  ``i``'s tail node with ``i+1``'s head node
            ``"user_define"``: We will give this option to the user. User can override this method
            to define your merge strategy.
        Returns
        -------
        joint_graph: GraphData
            The merged graph structure.
        """
        if cls.verbose > 0:
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
                if cls.verbose > 0:
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
                if node_attrs["head"] is True:
                    head = node_idx + node_idx_off
                if node_attrs["tail"] is True:
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
            if cls.verbose > 0:
                print("merged edges")
                print("src list:", src_list)
                print("tgt list:", tgt_list)
            g.add_edges(src_list, tgt_list)

        elif merge_strategy == "re_occurrences":
            token_list = [g.node_attributes[node_id]["token"] for node_id in g.node_attributes]
            token_count = dict(Counter(token_list))
            repeat_token = [key for key, value in token_count.items() if value > 1]
            re_occ_pair = []
            for token in repeat_token:
                re_occ_list = [i for i, v in enumerate(token_list) if v == token]
                re_occ_pair.extend(list(itertools.combinations(re_occ_list, 2)))
            for pair in re_occ_pair:
                g.add_edge(pair[0], pair[1])
                g.add_edge(pair[1], pair[0])
        else:
            raise NotImplementedError()

        for node_idx, node_attrs in g.node_attributes.items():
            node_attrs["head"] = node_idx == head_g
            node_attrs["tail"] = node_idx == tail_g

        if cls.verbose > 0:
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
        batch_graphdata = self.embedding_layer(batch_graphdata)
        return batch_graphdata

    def embedding(self, node_attributes, edge_attributes):
        node_emb, edge_emb = self.embedding_layer(node_attributes, edge_attributes)
        return node_emb, edge_emb
