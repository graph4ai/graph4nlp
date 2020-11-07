import json

from stanfordcorenlp import StanfordCoreNLP

from graph4nlp.pytorch.data.data import GraphData, to_batch
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from .base import StaticGraphConstructionBase
import dgl
import networkx as nx
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals
import numpy as np
import torch


class IEBasedGraphConstruction(StaticGraphConstructionBase):
    """
        Information Extraction based graph construction class

    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``single_token_item``, ``emb_strategy``, ``num_rnn_layers``, ``bert_model_name`` and ``bert_lower_case``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(self, embedding_style, vocab, hidden_size=300, fix_word_emb=True, fix_bert_emb=True, word_dropout=None,
                 rnn_dropout=None, device=None):
        super(IEBasedGraphConstruction, self).__init__(word_vocab=vocab,
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
    def parsing(cls, all_sent_triples_list, edge_strategy):
        """
        Parameters
        ----------
        all_sent_triples_list: list
        edge_strategy: str

        Returns
        -------
        parsed_results: dict
            parsed_results is an intermediate dict that contains all the information of
            the constructed IE graph for a piece of raw text input.

            `parsed_results['graph_content']` is a list of dict.

            Each dict in `parsed_results['graph_content']` contains information about a
            triple (src_ent, rel, tgt_ent).

            `parsed_results['graph_nodes']` contains all nodes in the KG graph.

            `parsed_results['node_num']` is the number of nodes in the KG graph.
        """

        parsed_results = {}
        parsed_results['graph_content'] = []
        graph_nodes = []
        for triple in all_sent_triples_list:
            if edge_strategy is None:
                if triple[0] not in graph_nodes:
                    graph_nodes.append(triple[0])

                if triple[2] not in graph_nodes:
                    graph_nodes.append(triple[2])

                triple_info = {'edge_tokens': triple[1],
                               'src': {
                                   'tokens': triple[0],
                                   'id': graph_nodes.index(triple[0])
                               },
                               'tgt': {
                                   'tokens': triple[2],
                                   'id': graph_nodes.index(triple[2])
                               }}
                if triple_info not in parsed_results['graph_content']:
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
                                       'tokens': triple[0],
                                       'id': graph_nodes.index(triple[0]),
                                       'type': 'ent_node'
                                   },
                                   'tgt': {
                                       'tokens': triple[1],
                                       'id': graph_nodes.index(triple[1]),
                                       'type': 'edge_node'
                                   }}

                triple_info_1_2 = {'edge_tokens': [],
                                   'src': {
                                       'tokens': triple[1],
                                       'id': graph_nodes.index(triple[1]),
                                       'type': 'edge_node'
                                   },
                                   'tgt': {
                                       'tokens': triple[2],
                                       'id': graph_nodes.index(triple[2]),
                                       'type': 'ent_node'
                                   }}

                if triple_info_0_1 not in parsed_results['graph_content']:
                    parsed_results['graph_content'].append(triple_info_0_1)
                if triple_info_1_2 not in parsed_results['graph_content']:
                    parsed_results['graph_content'].append(triple_info_1_2)
            else:
                raise NotImplementedError()

        parsed_results['node_num'] = len(graph_nodes)
        parsed_results['graph_nodes'] = graph_nodes

        return parsed_results

    @classmethod
    def topology(cls, raw_text_data, nlp_processor, processor_args, merge_strategy, edge_strategy, verbase=True):
        """
            Graph building method.

        Parameters
        ----------
        raw_text_data: str
            Raw text data, it can be multi-sentences.

        nlp_processor: StanfordCoreNLP
            NLP parsing tools

        merge_strategy: None or str, option=[None, "global", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``:  Do not add additional nodes and edges.

            ``global``: All subjects in extracted triples are connected by a "GLOBAL_NODE"
                        using a "global" edge

            ``"user_define"``: We will give this option to the user. User can override this method to define your merge
                               strategy.

        edge_strategy: None or str, option=[None, "as_node"]
            Strategy to process edge.
            ``None``: It will be the default option.
                      Edge information will be preserved in GraphDate.edge_attributes.
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node (``i``, ``k``) and (``k``, ``j``).
                         The ``type`` of original nodes will be set as ``ent_node``,
                         while the ``type`` of edge nodes is ``edge_node`.`

        Returns
        -------
        graph: GraphData
            The merged graph data-structure.
        """
        cls.verbase = verbase

        if isinstance(processor_args, list):
            props_coref = processor_args[0]
            props_openie = processor_args[1]
        else:
            raise RuntimeError('processor_args for IEBasedGraphConstruction shouble be a list of dict.')

        # Do coreference resolution on the whole 'raw_text_data'
        coref_json = nlp_processor.annotate(raw_text_data.strip(), properties=props_coref)
        from .utils import CORENLP_TIMEOUT_SIGNATURE
        if CORENLP_TIMEOUT_SIGNATURE in coref_json:
            raise TimeoutError('CoreNLP timed out at input: \n{}\n This item will be skipped. '
                               'Please check the input or change the timeout threshold.'.format(raw_text_data))

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
                if sbj+'<TSEP>'+rel not in all_sent_triples.keys():
                    all_sent_triples[sbj+'<TSEP>'+rel] = [sbj, rel, obj]
                else:
                    if len(obj)>len(all_sent_triples[sbj+'<TSEP>'+rel][2]):
                        all_sent_triples[sbj + '<TSEP>' + rel] = [sbj, rel, obj]

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
            if lst in all_sent_triples_list:
                all_sent_triples_list.remove(lst)

        global_triples = cls._graph_connect(all_sent_triples_list, merge_strategy)
        all_sent_triples_list.extend(global_triples)

        parsed_results = cls.parsing(all_sent_triples_list, edge_strategy)

        graph = cls._construct_static_graph(parsed_results, edge_strategy=edge_strategy)

        if cls.verbase:
            for info in parsed_results['graph_content']:
                print(info)
            try:
                print("is_connected="+str(nx.is_connected(nx.Graph(graph.to_dgl().to_networkx()))))
            except:
                print("is_connected=False")

        return graph


    def embedding(self, graph: GraphData):
        node_attributes = graph.node_attributes
        edge_attributes = graph.edge_attributes

        # Build embedding(initial feature vector) for graph nodes.
        # Each node may contains multiple tokens.
        node_idxs_list = []
        node_len_list = []
        for node_id, node_dict in node_attributes.items():
            node_word_idxs = []
            for token in node_dict['token'].split():
                node_word_idxs.append(self.vocab.getIndex(token))
            node_idxs_list.append(node_word_idxs)
            node_len_list.append(len(node_word_idxs))
        max_size = max(node_len_list)
        node_idxs_list = [x+[self.vocab.PAD]*(max_size-len(x)) for x in node_idxs_list]
        node_idxs_tensor = torch.LongTensor(node_idxs_list)
        # if self.embedding_layer.node_edge_emb_strategy == 'mean':
        #     node_len_tensor = torch.LongTensor(node_len_list).view(-1, 1)
        # else:
        node_len_tensor = torch.LongTensor(node_len_list)
        num_nodes = torch.LongTensor([len(node_len_list)])
        node_feat = self.embedding_layer(node_idxs_tensor, node_len_tensor, num_nodes)
        graph.node_features['node_feat'] = node_feat

        if 'token' in edge_attributes[0].keys():
            # If edge information is stored in `edge_attributes`,
            # build embedding(initial feature vector) for graph edges.
            # Each edge may contains multiple tokens.
            edge_idxs_list = []
            edge_len_list = []
            for edge_id, edge_dict in edge_attributes.items():
                edge_word_idxs = []
                for token in edge_dict['token']:
                    edge_word_idxs.append(self.vocab.getIndex(token))
                edge_idxs_list.append(edge_word_idxs)
                edge_len_list.append(len(edge_word_idxs))

            max_size = max(edge_len_list)
            edge_idxs_list = [x + [self.vocab.PAD] * (max_size - len(x)) for x in edge_idxs_list]
            edge_idxs_tensor = torch.LongTensor(edge_idxs_list)
            # if self.embedding_layer.node_edge_emb_strategy == 'mean':
            #     edge_len_tensor = torch.LongTensor(edge_len_list).view(-1, 1)
            # else:
            edge_len_tensor = torch.LongTensor(edge_len_list)
            num_edges = torch.LongTensor([len(edge_len_list)])
            edge_feat = self.embedding_layer(edge_idxs_tensor, edge_len_tensor, num_edges)
            graph.edge_features['edge_feat'] = edge_feat

        return graph


    @classmethod
    def _construct_static_graph(cls, parsed_object, edge_strategy=None):
        """

        Parameters
        ----------
        parsed_object: dict
            ``parsed_object`` contains all triples extracted from the raw_text_data.

        edge_strategy: None or str, option=[None, "as_node"]
            Strategy to process edge.
            ``None``: It will be the default option.
                      Edge information will be preserved in GraphDate.edge_attributes.
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node (``i``, ``k``) and (``k``, ``j``).
                         The ``type`` of original nodes will be set as ``ent_node``,
                         while the ``type`` of edge nodes is ``edge_node`.`

        Returns
        -------
        graph: GraphData
            graph structure for single sentence
        """
        ret_graph = GraphData()
        node_num = parsed_object["node_num"]
        ret_graph.add_nodes(node_num)
        for triple_info in parsed_object["graph_content"]:
            if edge_strategy is None:
                ret_graph.add_edge(triple_info["src"]['id'], triple_info['tgt']['id'])
                # eid = ret_graph.get_edge_num() - 1
                eids = ret_graph.edge_ids(triple_info["src"]['id'], triple_info['tgt']['id'])
                for eid in eids:
                    ret_graph.edge_attributes[eid]['token'] = triple_info['edge_tokens']
            elif edge_strategy == 'as_node':
                ret_graph.add_edge(triple_info["src"]['id'], triple_info['tgt']['id'])
            else:
                raise NotImplementedError()

            ret_graph.node_attributes[triple_info["src"]['id']]['token'] = triple_info["src"]['tokens']
            ret_graph.node_attributes[triple_info["tgt"]['id']]['token'] = triple_info['tgt']['tokens']
            if edge_strategy == 'as_node':
                ret_graph.node_attributes[triple_info["src"]['id']]['type'] = triple_info["src"]['type']
                ret_graph.node_attributes[triple_info["tgt"]['id']]['type'] = triple_info['tgt']['type']

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

        merge_strategy: None or str, option=[None, "global", "user_define"]
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

    def forward(self, batch_graphdata: list):
        node_size = []
        edge_size = []
        num_nodes = []
        num_edges = []

        token_id_max_len = 0
        edge_token_id_max_len = 0
        for g in batch_graphdata:
            token_id_len = g.node_features['token_id'].size()[1]
            if token_id_max_len < token_id_len:
                token_id_max_len = token_id_len
            if 'token' in batch_graphdata[0].edge_attributes[0].keys():
                edge_token_id_len = g.edge_features['token_id'].size()[1]
                if edge_token_id_max_len < edge_token_id_len:
                    edge_token_id_max_len = edge_token_id_len

        for g in batch_graphdata:
            g.node_features['token_id'] = torch.tensor(pad_2d_vals(np.array(g.node_features['token_id']),
                                                       g.node_features['token_id'].size()[0],
                                                       token_id_max_len),
                                                       dtype=torch.int64)
            g.node_features['token_id'] = g.node_features['token_id'].to(self.device)
            num_nodes.append(g.get_node_num())
            node_size.extend([len(g.node_attributes[i]['token_id']) for i in range(num_nodes[-1])])

            if 'token' in g.edge_attributes[0].keys():
                g.edge_features['token_id'] = torch.tensor(pad_2d_vals(np.array(g.edge_features['token_id']),
                                                                       g.edge_features['token_id'].size()[0],
                                                                       edge_token_id_max_len),
                                                           dtype=torch.int64)
                g.edge_features['token_id'] = g.edge_features['token_id'].to(self.device)
                num_edges.append(g.get_edge_num())
                edge_size.extend([len(g.edge_attributes[i]['token_id']) for i in range(num_edges[-1])])


        batch_gd = to_batch(batch_graphdata)
        node_size = torch.Tensor(node_size).to(self.device).int()
        num_nodes = torch.Tensor(num_nodes).to(self.device).int()
        node_emb = self.embedding_layer(batch_gd, node_size, num_nodes)
        batch_gd.node_features["node_feat"] = node_emb
        if edge_size != [] and num_edges != []:
            edge_size = torch.Tensor(edge_size).to(self.device).int()
            num_edges = torch.Tensor(num_edges).to(self.device).int()
            edge_emb = self.embedding_layer(batch_gd, edge_size, num_edges)
            batch_gd.edge_features["edge_feat"] = edge_emb

        return batch_gd
