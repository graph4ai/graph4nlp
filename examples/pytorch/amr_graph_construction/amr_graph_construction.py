import copy
import json
import spacy
import amrlib
from collections import defaultdict
from amrlib.alignments.faa_aligner import FAA_Aligner
from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.modules.graph_construction.base import StaticGraphConstructionBase


class AmrGraphConstruction(StaticGraphConstructionBase):
    """
        Dependency-parsing-tree based graph construction class

    Parameters
    ----------
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(
        self,
        vocab,
    ):
        super(AmrGraphConstruction, self).__init__()
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
    def parsing(cls, raw_text_data, nlp_processor, processor_args):
        """

        Parameters
        ----------
        raw_text_data: str

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
                    the entire word
                'id': int
                    the node token's id which will be used in GraphData
                'variable': str
                    the variable
                'type': int
                    the node type
                'sentence_id': int
                    the sentence id of the node
            "graph_content": list[dict]
                The list consisting edge information. Each edge is organized by a dict.
                "edge_type": str
                    The edge type token, eg: 'ARG1'
                'src': int
                    The source node ``id``
                'tgt': int.
                    The target node ``id``
                'sentence_id': int
                    The sentence id of the edge
            "sentence": str
                The original sentence of the amr graph.
            "mapping": dict[list]
                The mapping between sequence token index and node or edge index.
        """
        amrlib.setup_spacy_extension()
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(raw_text_data + ' .')
        parsed_results = []
        graphs = doc._.to_amr()
        st = []
        for ind, (graph, sentences) in enumerate(zip(graphs, doc.sents)):
            node_item = []
            parsed_sent = []
            st = []
            node_id = 0
            node2id = {}

            index = {}
            size_son = {}

            for line in graph.splitlines():
                if line[0] == '#':
                    continue
                l = line.strip().split()
                # add new node
                if line.find('/') != -1:
                    variable = l[l.index('/') - 1].strip('(')
                    concept = l[l.index('/') + 1].strip(')')
                    if '-' in concept:
                        concept = concept.split('-')[0]
                    assert concept is not ''
                    node = {
                        "variable": variable,
                        "id": node_id,
                        "token": concept,
                        "type": 4, # 4 for amr graph node
                        "sentence_id": ind,
                    }
                    node2id[variable] = node_id
                    nodeid_now = node_id
                    node_item.append(node)
                    node_id += 1
                else:
                    variable = l[-1].strip(')').strip('"')
                    if variable is '':
                        print(graph)
                        assert variable is not ''
                    if variable not in node2id:
                        node = {
                            "variable": None,
                            "id": node_id,
                            "token": variable, 
                            "type": 4, # 4 for amr graph node
                            "sentence_id": ind,
                        }
                        nodeid_now = node_id
                        node_item.append(node)
                        node_id += 1
                    else:
                        nodeid_now = node2id[variable]
                cnt = 0
                for c in line:
                    if c != ' ':
                        break
                    cnt += 1
                while len(st) and st[-1][0] >= cnt:
                    st.pop()
                nodeid_now = int(nodeid_now)
                # add new edge
                if line.find(':') != -1:
                    fa = st[-1][1]
                    pos = st[-1][2]
                    dep_info = {
                        "src": fa,
                        "tgt": nodeid_now,
                        "edge_type": l[0][1:],
                        "sentence_id": ind,
                    }
                    parsed_sent.append(dep_info)
                    for x in parsed_results:
                        if (x["src"] == fa and x["tgt"] == nodeid_now) \
                            or (x["src"] == nodeid_now and x["tgt"] == fa):
                            print(graphs)
                            assert 0
                    pos_now = pos + '.' + str(size_son[pos] + 1)
                    size_son[pos_now] = 0
                    index[pos_now] = nodeid_now
                    index[pos_now + '.r'] = len(parsed_sent) - 1
                    size_son[pos] += 1
                else:
                    pos_now = '1'
                    index[pos_now] = nodeid_now
                    size_son[pos_now] = 0
                
                # push the node to stack
                st.append((cnt, nodeid_now, pos_now))

            inference = FAA_Aligner()
            _, alignment_strings = inference.align_sents([sentences.text], [graph])
            alignment = alignment_strings[0].strip().split(' ')
            sentences_token = sentences.text.strip().split(' ')
            mapping = defaultdict(list)
            for relation in alignment:
                assert('-' in relation)
                src = relation.split('-')[0]
                tgt = relation.split('-')[1]
                assert(tgt in index)
                assert(int(src) <= len(sentences_token))
                if 'r' in tgt:
                    mapping[index[tgt]].append((int(src), "edge"))
                else:
                    mapping[index[tgt]].append((int(src), "node"))

            dep_dict = nlp_processor.annotate(sentences.text, properties=processor_args)
            pos_tag = [tokens["pos"] for tokens in dep_dict["sentences"][0]["tokens"]]
            entity_label = [tokens["ner"] for tokens in dep_dict["sentences"][0]["tokens"]]
            parsed_results.append(
                {"graph_content": parsed_sent, "node_content": node_item, "node_num": node_id, 
                    "sentence": sentences.text, "mapping": mapping, "pos_tag": pos_tag, "entity_label": entity_label}
            )   

        return parsed_results

    @classmethod
    def static_topology(
        cls,
        raw_text_data,
        merge_strategy=None,
        edge_strategy=None,
        nlp_processor=None,
        processor_args=None,
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
        verbose: int, default=0
            Whether to output log infors. Set 1 to output more infos.
        Returns
        -------
        joint_graph: GraphData
            The merged graph data-structure.
        """
        cls.verbose = verbose
        parsed_results = cls.parsing(raw_text_data, nlp_processor, processor_args)

        sub_graphs = []
        for parsed_sent in parsed_results:
            graph = cls._construct_static_graph(parsed_sent)
            sub_graphs.append(graph)
        joint_graph = cls._graph_connect(sub_graphs)
        return joint_graph

    @classmethod
    def _construct_static_graph(cls, parsed_object):
        """
            Build dependency-parsing-tree based graph for single sentence.

        Parameters
        ----------
        parsed_object: dict
            The parsing tree.

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
            ret_graph.node_attributes[node["id"]]["type"] = node["type"]
            ret_graph.node_attributes[node["id"]]["variable"] = node["variable"]
            ret_graph.node_attributes[node["id"]]["token"] = node["token"]
            ret_graph.node_attributes[node["id"]]["sentence_id"] = node["sentence_id"]
            ret_graph.node_attributes[node["id"]]["id"] = node["id"]
            ret_graph.node_attributes[node["id"]]["head"] = False
            ret_graph.node_attributes[node["id"]]["tail"] = False

        for dep_info in parsed_object["graph_content"]:
            ret_graph.add_edge(dep_info["src"], dep_info["tgt"])
            edge_idx = ret_graph.edge_ids(dep_info["src"], dep_info["tgt"])[0]
            ret_graph.edge_attributes[edge_idx]["token"] = dep_info["edge_type"]
            ret_graph.edge_attributes[edge_idx]["sentence_id"] = dep_info["sentence_id"]
        
        ret_graph.node_attributes[head_node]["head"] = True
        ret_graph.node_attributes[tail_node]["tail"] = True

        # add graph attributes
        ret_graph.graph_attributes["mapping"] = parsed_object["mapping"]
        ret_graph.graph_attributes["sentence"] = parsed_object["sentence"]
        ret_graph.graph_attributes["pos_tag"] = parsed_object["pos_tag"]
        ret_graph.graph_attributes["entity_label"] = parsed_object["entity_label"]

        return ret_graph

    @classmethod
    def _graph_connect(cls, nx_graph_list, merge_strategy=None):
        """
            This method will merge the sub-graphs into one graph.

        Parameters
        ----------
        nx_graph_list: list[GraphData]
            The list of all sub-graphs.

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
                print("sentence: {}".format(s_g.graph_attributes["sentence"]))
                print("mapping: {}".format(s_g.graph_attributes["mapping"]))
            print("*************************************")
        if len(nx_graph_list) == 0:
            raise RuntimeError("There is no graph needed to merge.")
        node_num_list = [s_g.get_node_num() for s_g in nx_graph_list]
        node_num = sum(node_num_list)
        g = GraphData()
        g.add_nodes(node_num)
        node_idx_off = 0
        # copy graph attributes
        g.graph_attributes["mapping"] = list()
        g.graph_attributes["sentence"] = list()
        g.graph_attributes["pos_tag"] = list()
        g.graph_attributes["entity_label"] = list()
        for graph in nx_graph_list:
            g.graph_attributes["mapping"].append(graph.graph_attributes["mapping"])
            g.graph_attributes["sentence"].append(graph.graph_attributes["sentence"])
            g.graph_attributes["pos_tag"].append(graph.graph_attributes["pos_tag"])
            g.graph_attributes["entity_label"].append(graph.graph_attributes["entity_label"])

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

            for key, value in enumerate(s_g.node_attributes):
                g.node_attributes[key + node_idx_off] = copy.deepcopy(value)
            node_idx_off += s_g.get_node_num()

        headtail_list = []

        head = -1
        tail = -1

        node_idx_off = 0
        for i in range(len(nx_graph_list)):
            for node_idx, node_attrs in enumerate(nx_graph_list[i].node_attributes):
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
        else:
            raise NotImplementedError()

        for node_idx, node_attrs in enumerate(g.node_attributes):
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
        raise RuntimeError("This interface is removed.")
