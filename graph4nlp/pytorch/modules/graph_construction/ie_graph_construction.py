import json

from stanfordcorenlp import StanfordCoreNLP

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from .base import StaticGraphConstructionBase

import networkx as nx


class IEBasedGraphConstruction(StaticGraphConstructionBase):
    """
        Information Extraction based graph construction class

    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(self, embedding_style, vocab, hidden_size=300, fix_word_emb=True, dropout=None, use_cuda=True):
        super(IEBasedGraphConstruction, self).__init__(word_vocab=vocab,
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

        merge_strategy: None or str, option=[None, "share_common_tokens", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``:   All subjects in extracted triples are connected by a "GLOBAL_NODE"
                        using a "global" edge
            ``"share_common_tokens"``:  The entity nodes share the same tokens are connected
                                        using a "COM" edge
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
        graph: GraphData
            The merged graph data-structure.
        """
        cls.verbase = 1

        # Do coreference resolution on the whole 'raw_text_data'
        props_coref = {
            'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
            "tokenize.options":
                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': False,
            'outputFormat': 'json'
        }
        coref_json = nlp_processor.annotate(raw_text_data.strip(), properties=props_coref)
        coref_dict = json.loads(coref_json)

        # Extract and preserve necessary parsing results from coref_dict['sentences']
        # sent_dict['tokenWords']: list of tokens in a sentence
        sentences = []
        for sent in coref_dict['sentences']:
            sent_dict = {}
            sent_dict['sentNum'] = sent['index']  # start from 0
            sent_dict['tokens'] = sent['tokens']
            sent_dict['tokenWords'] = [token['word'] for token in sent['tokens']]
            sent_dict['sentText'] = ' '.join(sent_dict['tokenWords'])
            sentences.append(sent_dict)

        for k, v in coref_dict['corefs'].items():
            # v is a list of dict, each dict contains a str
            # v[0] contains 'original entity str'
            # v[1:] contain 'pron strs' refers to 'original entity str'
            ent_text = v[0]['text']  # 'original entity str'
            if ',' in ent_text:
                # cut the 'original entity str' if it is too long
                ent_text = ent_text.split(',')[0].strip()
            ent_sentNum = v[0]['sentNum'] - 1  # the sentNum 'original entity str' appears in
            ent_startIndex = v[0]['startIndex'] - 1  # the startIndex 'original entity str' appears in
            ent_endIndex = v[0]['endIndex'] - 1  # the endIndex 'original entity str' appears in

            for pron in v[1:]:
                pron_text = pron['text']  # 'pron strs'
                if ent_text == pron_text or v[0]['text'] == pron_text:
                    continue
                pron_sentNum = pron['sentNum'] - 1  # the sentNum 'pron str' appears in
                pron_startIndex = pron['startIndex'] - 1
                pron_endIndex = pron['endIndex'] - 1

                # replace 'pron str' with 'original entity str'
                sentences[pron_sentNum]['tokenWords'][pron_startIndex] = ent_text
                for rm_idx in range(pron_startIndex+1, pron_endIndex):
                    sentences[pron_sentNum]['tokenWords'][rm_idx] = ""

        # build resolved text
        for sent_id, _ in enumerate(sentences):
            sentences[sent_id]['tokenWords'] = list(filter(lambda a: a != "", sentences[sent_id]['tokenWords']))
            sentences[sent_id]['resolvedText'] = ' '.join(sentences[sent_id]['tokenWords'])

        # use OpenIE to extract triples from resolvedText
        props_openie = {
            'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
            "tokenize.options":
                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': False,
            'outputFormat': 'json',
            "openie.triple.strict": "true"
        }

        all_sent_triples = {}
        for sent in sentences:
            resolved_sent = sent['resolvedText']
            openie_json = nlp_processor.annotate(resolved_sent.strip(), properties=props_openie)
            openie_dict = json.loads(openie_json)

            for triple_dict in openie_dict['sentences'][0]['openie']:
                sbj = triple_dict['subject']
                rel = triple_dict['relation']
                if rel in ['was', 'is', 'were', 'are']:
                    continue
                obj = triple_dict['object']

                # If two triples have the same subject and relation,
                # only preserve the one has longer object
                if sbj+'_'+rel not in all_sent_triples.keys():
                    all_sent_triples[sbj+'_'+rel] = [sbj, rel, obj]
                else:
                    if len(obj)>len(all_sent_triples[sbj+'_'+rel][2]):
                        all_sent_triples[sbj + '_' + rel] = [sbj, rel, obj]

        all_sent_triples_list = list(all_sent_triples.values())  # triples extracted from all sentences

        # remove similar triples
        triples_rm_list = []
        for i, lst_i in enumerate(all_sent_triples_list[:-1]):
            for j, lst_j in enumerate(all_sent_triples_list[i+1:]):
                str_i = ' '.join(lst_i)
                str_j = ' '.join(lst_j)
                if str_i in str_j or str_j in str_i or \
                        lst_i[0]+lst_i[2]==lst_j[0]+lst_j[2] or \
                        lst_i[1]+lst_i[2]==lst_j[1]+lst_j[2]:
                    if len(lst_i[1])>len(lst_j[1]):
                        triples_rm_list.append(lst_j)
                    else:
                        triples_rm_list.append(lst_i)

        for lst in triples_rm_list:
            all_sent_triples_list.remove(lst)

        global_triples = cls._graph_connect(all_sent_triples_list, merge_strategy)
        all_sent_triples_list.extend(global_triples)

        parsed_results = {}
        parsed_results['graph_content'] = []
        graph_nodes = []
        for triple in all_sent_triples_list:
            if edge_strategy is None or edge_strategy == "homogeneous":
                if triple[0] not in graph_nodes:
                    graph_nodes.append(triple[0])

                if triple[2] not in graph_nodes:
                    graph_nodes.append(triple[2])

                triple_info = {'edge_tokens': triple[1].split(),
                               'src': {
                                   'tokens': triple[0].split(),
                                   'id': graph_nodes.index(triple[0])
                               },
                               'tgt': {
                                   'tokens': triple[2].split(),
                                   'id': graph_nodes.index(triple[2])
                               }}
                parsed_results['graph_content'].append(triple_info)
            elif edge_strategy == "as_node":
                if triple[0] not in graph_nodes:
                    graph_nodes.append(triple[0])

                if triple[1] not in graph_nodes:
                    graph_nodes.append(triple[1])

                if triple[2] not in graph_nodes:
                    graph_nodes.append(triple[2])

                triple_info_0_1 = {'edge_tokens': [],
                               'src': {
                                   'tokens': triple[0].split(),
                                   'id': graph_nodes.index(triple[0]),
                                   'type': 'ent_node'
                               },
                               'tgt': {
                                   'tokens': triple[1].split(),
                                   'id': graph_nodes.index(triple[1]),
                                   'type': 'edge_node'
                               }}

                triple_info_1_2 = {'edge_tokens': [],
                                   'src': {
                                       'tokens': triple[1].split(),
                                       'id': graph_nodes.index(triple[1]),
                                       'type': 'edge_node'
                                   },
                                   'tgt': {
                                       'tokens': triple[2].split(),
                                       'id': graph_nodes.index(triple[2]),
                                       'type': 'ent_node'
                                   }}

                parsed_results['graph_content'].append(triple_info_0_1)
                parsed_results['graph_content'].append(triple_info_1_2)
            else:
                raise NotImplementedError()

        parsed_results['node_num'] = len(graph_nodes)
        parsed_results['graph_nodes'] = graph_nodes

        graph = cls._construct_static_graph(parsed_results, edge_strategy=edge_strategy)

        if cls.verbase:
            for info in parsed_results['graph_content']:
                print(info)
            print("is_connected="+str(nx.is_connected(nx.Graph(graph.to_dgl().to_networkx()))))

        return graph


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
            ``parsed_object`` contains all triples extracted from the raw_text_data.

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
        for triple_info in parsed_object["graph_content"]:
            if edge_strategy is None or edge_strategy == "homogeneous":
                ret_graph.add_edge(triple_info["src"]['id'], triple_info['tgt']['id'])
            elif edge_strategy == 'as_node':
                ret_graph.add_edge(triple_info["src"]['id'], triple_info['tgt']['id'])
            else:
                raise NotImplementedError()

            ret_graph.node_attributes[triple_info["src"]['id']]['token'] = triple_info["src"]['tokens']
            ret_graph.node_attributes[triple_info["tgt"]['id']]['token'] = triple_info['tgt']['tokens']
            if edge_strategy == 'as_node':
                ret_graph.node_attributes[triple_info["src"]['id']]['type'] = triple_info["src"]['type']
                ret_graph.node_attributes[triple_info["tgt"]['id']]['type'] = triple_info['tgt']['type']

            # TODO: add edge_attributes
        return ret_graph

    @classmethod
    def _graph_connect(cls, triple_list, merge_strategy=None):
        """
            This method will connect entities in the ``triple_list`` to ensure the graph
            is connected.

        Parameters
        ----------
        triple_list: list of [subject, relation, object]
            A list of all triples extracted from ``raw_text_data`` using coref and openie.

        merge_strategy: None or str, option=[None, "tailhead", "sequential", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``:  Do not add additional nodes and edges.

            ``global``: All subjects in extracted triples are connected by a "GLOBAL_NODE"
                        using a "global" edge

            ``"user_define"``: We will give this option to the user. User can override this method to define your merge
                               strategy.

        Returns
        -------
        global_triples: list of [subject, relation, object]
            The added triples using merge_strategy.
        """

        if merge_strategy == 'global':
            graph_nodes = []
            global_triples = []
            for triple in triple_list:
                if triple[0] not in graph_nodes:
                    graph_nodes.append(triple[0])
                    global_triples.append([triple[0], 'global', 'GLOBAL_NODE'])

                if triple[2] not in graph_nodes:
                    graph_nodes.append(triple[2])

            return global_triples
        elif merge_strategy == None:
            return []
        else:
            raise NotImplementedError()

        # ``"share_common_tokens"``:  The entity nodes share the same tokens are connected
        #                             using a "COM" edge
        # if merge_strategy=='share_common_tokens':
        #     common_tokens_triples = []
        #     for i, node_i in enumerate(graph_nodes[:-1]):
        #         for j, node_j in enumerate(graph_nodes[i+1:]):
        #             node_i_lst = node_i.split()
        #
        #             node_j_lst = node_j.split()
        #
        #             common_tokens = list(set(node_i_lst).intersection(set(node_j_lst)))
        #
        #             if common_tokens!=[]:
        #                 common_tokens_triples.append([node_i, 'COM', node_j])
        #
        #     global_triples.extend(common_tokens_triples)


    def forward(self, feat):
        pass
