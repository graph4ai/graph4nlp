from .base import StaticGraphConstructionBase
from stanfordcorenlp import StanfordCoreNLP
import json
from graph4nlp.pytorch.data.data import GraphData


class DependencyBasedGraphConstruction(StaticGraphConstructionBase):
    def __init__(self, embedding_style, vocab):
        super(DependencyBasedGraphConstruction, self).__init__(embedding_styles=embedding_style)
        self.vocab = vocab
        self.verbase = 1

    def topology(self, raw_text_data, nlp_processor, merge_strategy, edge_strategy):
        props = {
            'annotators': 'depparse',
            "tokenize.options":
                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': False,
            'outputFormat': 'json'
        }
        dep_json = nlp_processor.annotate(raw_text_data.lower().strip(), properties=props)
        dep_dict = json.loads(dep_json)
        parsed_results = []
        node_id = 0
        for s_id, s in enumerate(dep_dict["sentences"]):
            parsed_sent = []
            unique_hash = {}
            node_id = 0

            for dep in s["basicDependencies"]:
                if self.verbase > 0:
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
                        'word_id_in_sent': dep['governor'],
                        'id': unique_hash[(dep['governor'], dep['governorGloss'])],
                        "sent_id": s_id
                    },
                    'tgt': {
                        'token': dep['dependentGloss'],
                        'word_id_in_sent': dep['dependent'],
                        'id': unique_hash[(dep['dependent'], dep['dependentGloss'])],
                        "sent_id": s_id
                    }
                }
                if self.verbase > 0:
                    print(dep_info)
                parsed_sent.append(dep_info)
            if self.verbase > 0:
                print(node_id)
                print(len(parsed_sent))
            parsed_results.append({
                "graph_content": parsed_sent,
                "node_num": node_id
            })

        sub_graphs = []
        for sent_id, parsed_sent in enumerate(parsed_results):
            graph = self.construct_static_graph(parsed_sent, edge_strategy=None)
            sub_graphs.append(graph)
        joint_graph = self.graph_connect(sub_graphs, merge_strategy)
        return joint_graph

    def embedding(self, node_attributes, edge_attributes):
        pass

    def construct_static_graph(self, parsed_object, edge_strategy=None):
        ret_graph = GraphData()
        node_num = parsed_object["node_num"]
        ret_graph.add_nodes(node_num)
        for dep_info in parsed_object["graph_content"]:
            ret_graph.add_edge(dep_info["src"]['id'], dep_info['tgt']['id'])
            ret_graph.node_attributes[dep_info["src"]['id']]['token'] = dep_info["src"]['token']
            ret_graph.node_attributes[dep_info["tgt"]['id']]['token'] = dep_info['tgt']['token']
            ret_graph.node_attributes[dep_info["src"]['id']]['word_id_in_sent'] = dep_info["src"]['word_id_in_sent']
            ret_graph.node_attributes[dep_info["tgt"]['id']]['word_id_in_sent'] = dep_info['tgt']['word_id_in_sent']
            ret_graph.node_attributes[dep_info["src"]['id']]['type'] = 0
            ret_graph.node_attributes[dep_info["tgt"]['id']]['type'] = 0
            ret_graph.node_attributes[dep_info["src"]['id']]['sent_id'] = dep_info["src"]['sent_id']
            ret_graph.node_attributes[dep_info["tgt"]['id']]['sent_id'] = dep_info["src"]['sent_id']
            # TODO: add edge_attributes
        return ret_graph

    def graph_connect(self, nx_graph_list, merge_strategy=None):
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
            s_g_node_num = s_g.get_node_num()
            for i in range(s_g_node_num):
                g.node_attributes[node_idx_off + i]['token'] = s_g.node_attributes[i]['token']
                g.node_attributes[node_idx_off + i]['word_id_in_sent'] = s_g.node_attributes[i]['word_id_in_sent']
                g.node_attributes[node_idx_off + i]['type'] = s_g.node_attributes[i]['type']
                g.node_attributes[node_idx_off + i]['sent_id'] = s_g.node_attributes[i]['sent_id']
            node_idx_off += s_g.get_node_num()

        if merge_strategy is None or merge_strategy == "tailhead":
            headtail_list = []
            node_idx_off = 0
            for n_node in node_num_list:
                headtail_list.append((node_idx_off, node_idx_off + n_node - 1))
                node_idx_off += n_node
            src_list = []
            tgt_list = []
            for i in range(len(headtail_list) - 1):
                src_list.append(headtail_list[i][1])
                tgt_list.append(headtail_list[i+1][0])
            if self.verbase > 0:
                print("merged edges")
                print("src list:", src_list)
                print("tgt list:", tgt_list)
            g.add_edges(src_list, tgt_list)
        else:
            # TODO: add two merge strategy
            raise NotImplementedError()
        if self.verbase > 0:
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


