import copy
import json
from pythonds.basic.stack import Stack

from graph4nlp.pytorch.data.data import GraphData

from .base import StaticGraphConstructionBase
from .utils import CORENLP_TIMEOUT_SIGNATURE


class ConstituencyBasedGraphConstruction(StaticGraphConstructionBase):
    """
    Class for constituency graph construction.

    ...

    Attributes
    ----------
    embedding_styles : (dict)
        Specify embedding styles including ``single_token_item``, ``emb_strategy``,
        ``num_rnn_layers``, ``bert_model_name`` and ``bert_lower_case``.

    vocab: (set, optional)
        Vocabulary including all words appeared in graphs.

    Methods
    -------

    topology(raw_text_data, nlp_processor, merge_strategy=None, edge_strategy=None)
        Generate graph structure with nlp parser like ``CoreNLP`` etc.

    _construct_static_graph(parsed_object, sub_sentence_id, edge_strategy=None)
        Construct a single static graph from a single sentence,
        to be called by ``topology`` function.

    _graph_connect(nx_graph_list, merge_strategy=None)
        Construct a merged graph from a list of graphs, to be called by ``topology`` function.

    embedding(node_attributes, edge_attributes)
        Generate node/edge embeddings from node/edge attributes through an embedding layer.

    forward(raw_text_data, nlp_parser)
        Generate graph topology and embeddings.
    """

    def __init__(self, vocab):
        super(ConstituencyBasedGraphConstruction, self).__init__()
        self.vocab = vocab

    @classmethod
    def parsing(cls, raw_text_data, nlp_processor, processor_args):
        """
        Parameters
        ----------
        raw_text_data: str
        nlp_processor: StanfordCoreNLP
        processor_args: json config for constituency graph construction
        """
        output = nlp_processor.annotate(
            raw_text_data.strip().replace("(", "<LB>").replace(")", "<RB>"),
            properties=processor_args,
        )
        if CORENLP_TIMEOUT_SIGNATURE in output:
            raise TimeoutError(
                "CoreNLP timed out at input: \n{}\n This item will be skipped. "
                "Please check the input or change the timeout threshold.".format(raw_text_data)
            )
        parsed_output = json.loads(output)["sentences"]
        return parsed_output

    @classmethod
    def static_topology(
        cls,
        raw_text_data,
        nlp_processor,
        processor_args,
        merge_strategy=None,
        edge_strategy=None,
        sequential_link=3,
        top_down=False,
        prune=2,
        verbose=True,
    ):
        """topology This function generate a graph strcuture from a raw text data.

        Parameters
        ----------
        raw_text_data : string
            A string to be used to construct a static graph, can be composed of multiple strings

        nlp_processor : object
            A parser used to parse sentence string to parsing trees like dependency parsing tree
            or constituency parsing tree

        merge_strategy : None or str, option=[None, "tailhead", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``: It will be the default option. We will do as ``"tailhead"``.
            ``"tailhead"``: Link the sub-graph  ``i``'s tail node with ``i+1``'s head node
            ``"user_define"``: We will give this option to the user. User can override the
                                method ``_graph_connnect`` to define your merge strategy.

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
                         we will insert a node ``k`` into the graph and link
                         node (``i``, ``k``) and (``k``, ``j``). It is not implemented yet.

        sequential_link : int, option=[0,1,2,3]
            Strategy to add sequential links between word nodes.
            ``0``: Do not add sequential links.
            ``1``: Add unidirectional links.
            ``2``: Add bidirectional links.
            ``3``: Do not add sequential links inside each sentence and add bidirectional links
                   between adjacent sentences.

        top_down : bool
            If true, edges in constituency tree are from root nodes to leaf nodes. Otherwise,
            from leaf nodes to root nodes.

        prune : int, option=[0,1,2]
            Strategies for pruning constituency trees
            ``0``: No pruning.
            ``1``: Prune pos nodes.
            ``2``: Prune nodes with both in-degree and out-degree of 1.

        verbose : bool
            A boolean option to decide whether to print out the graph construction process.

        Returns
        -------
        GraphData
            A customized graph data structure
        """
        output_graph_list = []
        parsed_output = cls.parsing(raw_text_data, nlp_processor, processor_args)
        if prune == 0:
            add_pos_nodes = True
            cut_line_node = False
        elif prune == 1:
            add_pos_nodes = False
            cut_line_node = False
        elif prune == 2:
            add_pos_nodes = False
            cut_line_node = True
        else:
            raise ValueError("``prune`` should be chosen from [0,1,2].")

        if sequential_link == 0:
            seq_link = False
            biseq_link = False
        elif sequential_link == 1:
            seq_link = True
            biseq_link = False
        elif sequential_link == 2:
            seq_link = True
            biseq_link = True
        elif sequential_link == 3:
            seq_link = False
            biseq_link = True
        else:
            raise ValueError("``sequential_link`` should be chosen from [0,1,2,3].")

        if cut_line_node and sequential_link == 1:
            raise ValueError(
                "``cut_line_node`` should not be used when edges\
                 between word nodes are unidirectional."
            )

        for index in range(len(parsed_output)):
            # print(parsed_output[index]['parse'])
            output_graph_list.append(
                cls._construct_static_graph(
                    parsed_output[index],
                    index,
                    sequential_link=seq_link,
                    bisequential_link=biseq_link,
                    top_down=top_down,
                    add_pos_node=add_pos_nodes,
                    cut_line_node=cut_line_node,
                )
            )
        ret_graph = cls._graph_connect(output_graph_list, bisequential_link=biseq_link)
        # for edge_item in ret_graph.get_all_edges():
        #     print(ret_graph.node_attributes[edge_item[0]]['token'], "->",
        #       ret_graph.node_attributes[edge_item[1]]['token'])
        if verbose:
            print("--------------------------------------")
            for _edge in ret_graph.get_all_edges():
                print(
                    ret_graph.node_attributes[_edge[0]]["token"],
                    "\t---\t",
                    ret_graph.node_attributes[_edge[1]]["token"],
                )
            print("--------------------------------------")
        return ret_graph

    @classmethod
    def _construct_static_graph(
        cls,
        parsed_object,
        sub_sentence_id,
        edge_strategy=None,
        sequential_link=False,
        bisequential_link=True,
        top_down=False,
        add_pos_node=False,
        cut_line_node=True,
    ):
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
            Edge direction between nodes in constituency tree, if ``True``,
            add edges from top to down

        add_pos_node : bool
            Add part-of-speech nodes or not

        cut_line_node : bool
            Cut nodes which are organized in "line" format (degree = 2)
        """
        assert (add_pos_node & cut_line_node) is False
        parsed_sentence_data = parsed_object["parse"]
        for punc in [u"(", u")"]:
            parsed_sentence_data = parsed_sentence_data.replace(punc, " " + punc + " ")
        parse_list = (parsed_sentence_data.strip()).split()
        # cut root node
        if parse_list[0] == "(" and parse_list[1] == "ROOT":
            parse_list = parse_list[2:-1]
        # transform '.' to 'period'
        for index in range(len(parse_list)):
            if index <= len(parse_list) - 2:
                if parse_list[index] == "." and parse_list[index + 1] == ".":
                    parse_list[index] = "period"
        res_graph = GraphData()
        pstack = Stack()
        idx = 0
        cnt_word_node = -1
        while idx < len(parse_list):
            if parse_list[idx] == "(":
                res_graph.add_nodes(1)
                # res_graph.node_attributes[res_graph.get_node_num(
                # ) - 1] = {'token': parse_list[idx + 1], 'type': 1,
                # 'position_id': None, 'sentence_id': sub_sentence_id,
                #           'tail': False, 'head': False}
                res_graph.node_attributes[res_graph.get_node_num() - 1] = {
                    "token": parse_list[idx + 1],
                    "type": 1,
                    "position_id": None,
                    "sentence_id": sub_sentence_id,
                    "tail": False,
                    "head": False,
                }
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
            elif parse_list[idx] == ")":
                pstack.pop()
            elif parse_list[idx + 1] == u")" and parse_list[idx] != u")":
                cnt_word_node += 1
                if add_pos_node:
                    res_graph.add_nodes(1)
                else:
                    assert res_graph.node_attributes[res_graph.get_node_num() - 1]["type"] != 0
                res_graph.node_attributes[res_graph.get_node_num() - 1] = {
                    "token": parse_list[idx],
                    "type": 0,
                    "position_id": cnt_word_node,
                    "sentence_id": sub_sentence_id,
                    "tail": False,
                    "head": False,
                }
                node_1 = pstack.pop()
                if add_pos_node:
                    assert node_1 != res_graph.get_node_num() - 1
                    if top_down:
                        res_graph.add_edge(node_1, res_graph.get_node_num() - 1)
                    else:
                        res_graph.add_edge(res_graph.get_node_num() - 1, node_1)
                else:
                    assert node_1 == res_graph.get_node_num() - 1
                pstack.push(node_1)
            idx += 1

        if sequential_link:
            _len_single_graph = res_graph.get_node_num()
            _cnt_node = 0
            while True:
                node_1_idx = -1
                node_2_idx = -1
                for idx_seq, node_attr in enumerate(res_graph.node_attributes):
                    if node_attr["position_id"] == _cnt_node:
                        node_1_idx = idx_seq
                    elif node_attr["position_id"] == _cnt_node + 1:
                        node_2_idx = idx_seq
                if node_1_idx != -1 and node_2_idx != -1:
                    res_graph.add_edge(node_1_idx, node_2_idx)
                    if bisequential_link:
                        res_graph.add_edge(node_2_idx, node_1_idx)
                if _cnt_node >= _len_single_graph:
                    break
                _cnt_node += 1

        max_pos = 0
        for n in res_graph.node_attributes:
            if n["type"] == 0 and n["position_id"] > max_pos:
                max_pos = n["position_id"]

        min_pos = 99999
        for n in res_graph.node_attributes:
            if n["type"] == 0 and n["position_id"] < min_pos:
                min_pos = n["position_id"]

        for n in res_graph.node_attributes:
            if n["type"] == 0 and n["position_id"] == max_pos:
                n["tail"] = True
            if n["type"] == 0 and n["position_id"] == min_pos:
                n["head"] = True
        # print(res_graph.edges)
        if cut_line_node:
            res_graph = cls._cut_line_node(res_graph)
        return res_graph

    @classmethod
    def _graph_connect(
        cls, graph_list, merge_strategy=None, bisequential_link=True, reformalize=True
    ):
        """
        Parameters
        ----------
        graph_list : list
            A graph list to be merged

        bisequential_link : bool
            whether add bi-direnctional links between word nodes

        reformalize : bool
            If true, separate word nodes and non-terminal nodes in ``graph.node_attributes``
            and put word nodes in the front position

        Returns
        -------
        GraphData
            A customized graph data structure
        """
        _len_graph_ = len(graph_list)
        # if only one sentence included, we do not need to merge the graph.
        if _len_graph_ > 1:
            merged_graph = GraphData()
            for index in range(_len_graph_):
                # len_merged_graph = merged_graph.get_node_num()

                graph_i_nodes_attributes = []
                for list_item in graph_list[index].node_attributes:
                    graph_i_nodes_attributes.append(list_item)
                merged_graph.node_attributes.extend(graph_i_nodes_attributes)

            _cnt_edge_num = 0
            for g_idx in range(_len_graph_):
                tmp_edges = graph_list[g_idx].get_all_edges()
                current_node_num = graph_list[g_idx].get_node_num()
                for _edge in tmp_edges:
                    merged_graph.add_edge(_edge[0] + _cnt_edge_num, _edge[1] + _cnt_edge_num)
                _cnt_edge_num += current_node_num

            for index in range(_len_graph_ - 1):
                # get head node
                head_node = -1
                for idx, node_attr_item in enumerate(merged_graph.node_attributes):
                    if (
                        node_attr_item["sentence_id"] == index + 1
                        and node_attr_item["type"] == 0
                        and node_attr_item["head"] is True
                    ):
                        head_node = idx
                # get tail node
                tail_node = -1
                for idx, node_attr_item in enumerate(merged_graph.node_attributes):
                    if (
                        node_attr_item["sentence_id"] == index
                        and node_attr_item["type"] == 0
                        and node_attr_item["tail"] is True
                    ):
                        tail_node = idx
                merged_graph.add_edge(tail_node, head_node)
                if bisequential_link:
                    merged_graph.add_edge(head_node, tail_node)
        else:
            merged_graph = graph_list[0]
        if reformalize:
            new_list_for_nodes = copy.deepcopy(merged_graph.node_attributes)

            list_for_word_nodes = []
            list_for_non_ternimal_nodes = []
            for i_0, i_1 in enumerate(new_list_for_nodes):
                if i_1["type"] == 0:
                    list_for_word_nodes.append((i_0, i_1))
                else:
                    list_for_non_ternimal_nodes.append((i_0, i_1))

            node_id_map = {}
            cnt = 0
            merged_graph.node_attributes.clear()
            for item in list_for_word_nodes:
                merged_graph.node_attributes.append(item[1])
                node_id_map[item[0]] = cnt
                cnt += 1
            for item in list_for_non_ternimal_nodes:
                merged_graph.node_attributes.append(item[1])
                node_id_map[item[0]] = cnt
                cnt += 1
            for i in range(merged_graph.get_edge_num()):
                merged_graph._edge_indices.src[i] = node_id_map[merged_graph._edge_indices.src[i]]
                merged_graph._edge_indices.tgt[i] = node_id_map[merged_graph._edge_indices.tgt[i]]
        return merged_graph

    @classmethod
    def _cut_line_node(cls, input_graph: GraphData):
        idx_to_be_deleted = []
        new_edges = []
        for idx, _ in enumerate(input_graph.node_attributes):
            edge_arr = input_graph.get_all_edges()
            cnt_in = 0
            cnt_out = 0
            for e in edge_arr:
                if idx == e[0]:
                    cnt_out += 1
                    out_ = e[1]
                if idx == e[1]:
                    cnt_in += 1
                    in_ = e[0]
            if cnt_in == 1 and cnt_out == 1:
                idx_to_be_deleted.append(idx)
                new_edges.append((in_, out_))
        if len(idx_to_be_deleted) == 0:
            return input_graph
        res_graph = GraphData()
        id_map = {}
        cnt_node = 0
        for idx, n in enumerate(input_graph.node_attributes):
            if idx not in idx_to_be_deleted:
                res_graph.add_nodes(1)
                res_graph.node_attributes[res_graph.get_node_num() - 1] = n
                id_map[idx] = cnt_node
                cnt_node += 1
            else:
                id_map[idx] = -1
        for edge_arr in input_graph.get_all_edges() + new_edges:
            if (edge_arr[0] not in idx_to_be_deleted) and (edge_arr[1] not in idx_to_be_deleted):
                res_graph.add_edge(id_map[edge_arr[0]], id_map[edge_arr[1]])
        return res_graph

    def forward(self, batch_graphdata: list):
        raise RuntimeError("This interface is removed.")
