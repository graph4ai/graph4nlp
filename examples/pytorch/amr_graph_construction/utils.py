import copy
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.data.dataset import DataItem, Text2TreeDataset

from graph4nlp.pytorch.datasets.mawps import MawpsDatasetForTree
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.utils.tree_utils import Tree

from examples.pytorch.rgcn.rgcn import RGCN

warnings.filterwarnings("ignore")

class AMRDataItem(DataItem):
    def __init__(self, input_text, output_text, output_tree, tokenizer, share_vocab=True):
        super(AMRDataItem, self).__init__(input_text, tokenizer)
        self.output_text = output_text
        self.share_vocab = share_vocab
        self.output_tree = output_tree

    def extract(self):
        """
        Returns
        -------
        Input tokens and output tokens
        """
        g: GraphData = self.graph

        input_tokens = []
        for i in range(g.get_node_num()):
            tokenized_token = self.tokenizer(g.node_attributes[i]["token"])
            input_tokens.extend(tokenized_token)
        
        for s in g.graph_attributes["sentence"]:
            input_tokens.extend(s.strip().split(" "))

        output_tokens = self.tokenizer(self.output_text)

        return input_tokens, output_tokens

class RGCNGraph2Tree(Graph2Tree):
    def __init__(
        self,
        vocab_model,
        edge_vocab,
        embedding_style,
        graph_construction_name,
        # embedding
        emb_input_size,
        emb_hidden_size,
        emb_word_dropout,
        emb_rnn_dropout,
        emb_fix_word_emb,
        emb_fix_bert_emb,
        # gnn
        gnn,
        gnn_num_layers,
        gnn_direction_option,
        gnn_input_size,
        gnn_hidden_size,
        gnn_output_size,
        gnn_feat_drop,
        gnn_attn_drop,
        # decoder
        dec_use_copy,
        dec_hidden_size,
        dec_dropout,
        dec_teacher_forcing_rate,
        dec_max_decoder_step,
        dec_max_tree_depth,
        dec_attention_type,
        dec_use_sibling,
        # optional
        criterion=None,
        share_vocab=False,
        **kwargs
    ):
        super(RGCNGraph2Tree, self).__init__(
            vocab_model=vocab_model,
            embedding_style=embedding_style,
            graph_construction_name=graph_construction_name,
            # embedding
            emb_input_size=emb_input_size,
            emb_hidden_size=emb_hidden_size,
            emb_word_dropout=emb_word_dropout,
            emb_rnn_dropout=emb_rnn_dropout,
            emb_fix_word_emb=emb_fix_word_emb,
            emb_fix_bert_emb=emb_fix_bert_emb,
            # gnn
            gnn=gnn,
            gnn_num_layers=gnn_num_layers,
            gnn_direction_option=gnn_direction_option,
            gnn_input_size=gnn_input_size,
            gnn_hidden_size=gnn_hidden_size,
            gnn_output_size=gnn_output_size,
            gnn_feat_drop=gnn_feat_drop,
            gnn_attn_drop=gnn_attn_drop,
            # decoder
            dec_use_copy=dec_use_copy,
            dec_hidden_size=dec_hidden_size,
            dec_dropout=dec_dropout,
            dec_teacher_forcing_rate=dec_teacher_forcing_rate,
            dec_max_decoder_step=dec_max_decoder_step,
            dec_max_tree_depth=dec_max_tree_depth,
            dec_attention_type=dec_attention_type,
            dec_use_sibling=dec_use_sibling,
            # optional
            criterion=criterion,
            share_vocab=share_vocab,
            **kwargs
        )
        self.edge_vocab = edge_vocab

    def _build_gnn_encoder(
        self,
        gnn,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        direction_option,
        feats_dropout,
        gnn_heads=None,
        gnn_residual=True,
        gnn_attn_dropout=0.0,
        gnn_activation=F.relu,  # gat
        gnn_bias=True,
        gnn_allow_zero_in_degree=True,
        gnn_norm="both",
        gnn_weight=True,
        gnn_use_edge_weight=False,
        gnn_gcn_norm="both",  # gcn
        gnn_n_etypes=1,  # ggnn
        gnn_aggregator_type="lstm",  # graphsage
        **kwargs
    ):
        if gnn == "rgcn":
            self.gnn_encoder = RGCN(
                num_layers,
                input_size,
                hidden_size,
                output_size,
                num_rels=77,
                num_bases=4,
                gpu=0,
            )
        else:
            raise NotImplementedError()
    
    @classmethod
    def from_args(cls, opt, vocab_model, edge_vocab):
        """
            The function for building ``Graph2Tree`` model.
        Parameters
        ----------
        opt: dict
            The configuration dict. It should has the same hierarchy and keys as the template.
        vocab_model: VocabModel
            The vocabulary.

        Returns
        -------
        model: Graph2Tree
        """
        initializer_args = cls._get_node_initializer_params(opt)
        gnn_args = cls._get_gnn_params(opt)
        dec_args = cls._get_decoder_params(opt)

        args = copy.deepcopy(initializer_args)
        args.update(gnn_args)
        args.update(dec_args)
        args["share_vocab"] = opt["model_args"]["graph_construction_args"][
            "graph_construction_share"
        ][
            "share_vocab"
        ]  # noqa
        return cls(vocab_model=vocab_model, edge_vocab=edge_vocab, **args)

class InferenceText2TreeDataset(Text2TreeDataset):
    def __init__(
        self,
        graph_construction_name: str,
        root_dir: str = None,
        static_or_dynamic: str = "static",
        topology_builder = None,
        topology_subdir: str = None,
        dynamic_init_graph_name: str = None,
        dynamic_init_topology_builder = None,
        dynamic_init_topology_aux_args=None,
        share_vocab=True,
        dataitem=None,
        init_edge_vocab=True,
        **kwargs,
    ):
        super(InferenceText2TreeDataset, self).__init__(
            root_dir=root_dir,
            graph_construction_name=graph_construction_name,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            static_or_dynamic=static_or_dynamic,
            share_vocab=share_vocab,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            init_edge_vocab=init_edge_vocab,
            **kwargs,
        )
        self.data_item_type = dataitem

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by
        each individual task-specific base class. Returns all the indices of data items
        in this file w.r.t. the whole dataset.

        For Text2TreeDataset, the format of the input file should contain lines of input,
        each line representing one record of data. The input and output is separated by
        a tab(\t).

        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )

        DataItem: input_text="list job use languageid0", output_text="job ( ANS ) ,
        language ( ANS , languageid0 )"

        Parameters
        ----------
        file_path: str
            The path of the input file.

        Returns
        -------
        list
            The indices of data items in the file w.r.t. the whole dataset.
        """
        data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                input, output = line.split("\t")
                data_item = self.data_item_type(
                    input_text=input,
                    output_text=output,
                    output_tree=None,
                    tokenizer=self.tokenizer,
                    share_vocab=self.share_vocab,
                )
                data.append(data_item)
        return data

    def vectorization(self, data_items):
        """For tree decoder we also need the vectorize the tree output."""
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]["token"]
                node_token_id = self.vocab_model.in_word_vocab.get_symbol_idx(node_token)
                graph.node_attributes[node_idx]["token_id"] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix

            token_matrix = []
            if self.init_edge_vocab:
                for edge_idx in range(graph.get_edge_num()):
                    if "token" in graph.edge_attributes[edge_idx]:
                        edge_token = graph.edge_attributes[edge_idx]["token"]
                    else:
                        edge_token = ""
                    edge_token_id = self.edge_vocab[edge_token]
                    graph.edge_attributes[edge_idx]["token_id"] = edge_token_id
                    token_matrix.append([edge_token_id])
                token_matrix = torch.tensor(token_matrix, dtype=torch.long)
                graph.edge_features["token_id"] = token_matrix
            
            if "pos_tag" in graph.graph_attributes:
                pos_vocab = [".", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
                pos_map = {pos: i for i, pos in enumerate(pos_vocab)}
                maxlen = max(len(pos_tag) for pos_tag in graph.graph_attributes["pos_tag"])
                pos_token_id = torch.zeros(len(graph.graph_attributes["pos_tag"]), maxlen, dtype=torch.long)
                for i, sentence_token in enumerate(graph.graph_attributes["pos_tag"]):
                    for j, token in enumerate(sentence_token):
                        if token in pos_map:
                            pos_token_id[i][j] = pos_map[token]
                        else:
                            print('pos_tag', token)
                graph.graph_attributes["pos_tag_id"] = pos_token_id

            if "entity_label" in graph.graph_attributes:
                entity_label = ["O", "PERSON", "LOCATION", "ORGANIZATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME", "DURATION", "SET", "EMAIL", "URL", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION", "TITLE", "IDEOLOGY", "CRIMINAL_CHARGE", "CAUSE_OF_DEATH", "HANDLE"]
                entity_map = {entity: i for i, entity in enumerate(entity_label)}
                maxlen = max(len(entity_tag) for entity_tag in graph.graph_attributes["entity_label"])
                entity_token_id = torch.zeros(len(graph.graph_attributes["entity_label"]), maxlen, dtype=torch.long)
                for i, sentence_token in enumerate(graph.graph_attributes["entity_label"]):
                    for j, token in enumerate(sentence_token):
                        if token in entity_map:
                            entity_token_id[i][j] = entity_map[token]
                        else:
                            print('entity_label', token)
                graph.graph_attributes["entity_label_id"] = entity_token_id

            if "sentence" in graph.graph_attributes:
                maxlen = max(len(sentence.strip().split()) for sentence in graph.graph_attributes["sentence"])
                seq_token_id = torch.zeros(len(graph.graph_attributes["sentence"]), maxlen, dtype=torch.long)
                for i, sentence in enumerate(graph.graph_attributes["sentence"]):
                    sentence_token = sentence.strip().split()
                    for j, token in enumerate(sentence_token):
                        seq_token_id[i][j] = self.vocab_model.in_word_vocab.get_symbol_idx(token)
                graph.graph_attributes["sentence_id"] = seq_token_id

            tgt = item.output_text
            tgt_list = self.vocab_model.out_word_vocab.get_symbol_idx_for_list(tgt.split())
            output_tree = Tree.convert_to_tree(
                tgt_list, 0, len(tgt_list), self.vocab_model.out_word_vocab
            )
            item.output_tree = output_tree

    def _vectorize_one_dataitem(cls, data_item, vocab_model, use_ie=False, edge_vocab=None):
        item = deepcopy(data_item)
        graph: GraphData = item.graph
        token_matrix = []
        for node_idx in range(graph.get_node_num()):
            node_token = graph.node_attributes[node_idx]["token"]
            node_token_id = vocab_model.in_word_vocab.get_symbol_idx(node_token)
            graph.node_attributes[node_idx]["token_id"] = node_token_id
            token_matrix.append([node_token_id])
        token_matrix = torch.tensor(token_matrix, dtype=torch.long)
        graph.node_features["token_id"] = token_matrix
        if edge_vocab is not None:
            token_matrix = []
            for edge_idx in range(graph.get_edge_num()):
                if "token" in graph.edge_attributes[edge_idx]:
                    edge_token = graph.edge_attributes[edge_idx]["token"]
                else:
                    edge_token = ""
                edge_token_id = edge_vocab[edge_token]
                graph.edge_attributes[edge_idx]["token_id"] = edge_token_id
                token_matrix.append([edge_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.edge_features["token_id"] = token_matrix

            
            if "pos_tag" in graph.graph_attributes:
                pos_vocab = [".", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
                pos_map = {pos: i for i, pos in enumerate(pos_vocab)}
                maxlen = max(len(pos_tag) for pos_tag in graph.graph_attributes["pos_tag"])
                pos_token_id = torch.zeros(len(graph.graph_attributes["pos_tag"]), maxlen, dtype=torch.long)
                for i, sentence_token in enumerate(graph.graph_attributes["pos_tag"]):
                    for j, token in enumerate(sentence_token):
                        if token in pos_map:
                            pos_token_id[i][j] = pos_map[token]
                        else:
                            print('pos_tag', token)
                graph.graph_attributes["pos_tag_id"] = pos_token_id

            if "entity_label" in graph.graph_attributes:
                entity_label = ["O", "PERSON", "LOCATION", "ORGANIZATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME", "DURATION", "SET", "EMAIL", "URL", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION", "TITLE", "IDEOLOGY", "CRIMINAL_CHARGE", "CAUSE_OF_DEATH", "HANDLE"]
                entity_map = {entity: i for i, entity in enumerate(entity_label)}
                maxlen = max(len(entity_tag) for entity_tag in graph.graph_attributes["entity_label"])
                entity_token_id = torch.zeros(len(graph.graph_attributes["entity_label"]), maxlen, dtype=torch.long)
                for i, sentence_token in enumerate(graph.graph_attributes["entity_label"]):
                    for j, token in enumerate(sentence_token):
                        if token in entity_map:
                            entity_token_id[i][j] = entity_map[token]
                        else:
                            print('entity_label', token)
                graph.graph_attributes["entity_label_id"] = entity_token_id

            if "sentence" in graph.graph_attributes:
                maxlen = max(len(sentence.strip().split()) for sentence in graph.graph_attributes["sentence"])
                seq_token_id = torch.zeros(len(graph.graph_attributes["sentence"]), maxlen, dtype=torch.long)
                for i, sentence in enumerate(graph.graph_attributes["sentence"]):
                    sentence_token = sentence.strip().split()
                    for j, token in enumerate(sentence_token):
                        seq_token_id[i][j] = vocab_model.in_word_vocab.get_symbol_idx(token)
                graph.graph_attributes["sentence_id"] = seq_token_id

        if isinstance(item.output_text, str):
            tgt = item.output_text
            tgt_list = vocab_model.out_word_vocab.get_symbol_idx_for_list(tgt.split())
            output_tree = Tree.convert_to_tree(
                tgt_list, 0, len(tgt_list), vocab_model.out_word_vocab
            )
            item.output_tree = output_tree
        return item


class EdgeText2TreeDataset(MawpsDatasetForTree):
    def __init__(
        self,
        root_dir,
        # topology_builder,
        topology_subdir,
        graph_construction_name,
        static_or_dynamic="static",
        topology_builder=None,
        merge_strategy="tailhead",
        edge_strategy=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        nlp_processor_args=None,
        #  pretrained_word_emb_file=None,
        pretrained_word_emb_name="6B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        val_split_ratio=0,
        word_emb_size=300,
        share_vocab=True,
        enc_emb_size=300,
        dec_emb_size=300,
        min_word_vocab_freq=1,
        max_word_vocab_size=100000,
        for_inference=False,
        reused_vocab_model=None,
        dataitem=None,
        init_edge_vocab=True,
    ):
        """
        Parameters
        ----------
        root_dir: str
            The path of dataset.
        graph_name: str
            The name of graph construction method. E.g., "dependency".
            Note that if it is in the provided graph names (i.e., "dependency", \
                "constituency", "ie", "node_emb", "node_emb_refine"), the following \
                parameters are set by default and users can't modify them:
                1. ``topology_builder``
                2. ``static_or_dynamic``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        topology_builder: GraphConstructionBase
            The graph construction class.
        topology_subdir: str
            The directory name of processed path.
        static_or_dynamic: str, default='static'
            The graph type. Expected in ('static', 'dynamic')
        edge_strategy: str, default=None
            The edge strategy. Expected in (None, 'homogeneous', 'as_node').
            If set `None`, it will be 'homogeneous'.
        merge_strategy: str, default=None
            The strategy to merge sub-graphs. Expected in (None, 'tailhead', 'user_define').
            If set `None`, it will be 'tailhead'.
        share_vocab: bool, default=False
            Whether to share the input vocabulary with the output vocabulary.
        dynamic_init_graph_name: str, default=None
            The graph name of the initial graph. Expected in (None, "line", "dependency", \
                "constituency").
            Note that if it is in the provided graph names (i.e., "line", "dependency", \
                "constituency"), the following parameters are set by default and users \
                can't modify them:
                1. ``dynamic_init_topology_builder``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        dynamic_init_topology_builder: GraphConstructionBase
            The graph construction class.
        dynamic_init_topology_aux_args: None,
            TBD.
        """
        # Initialize the dataset. If the preprocessed files are not found,
        # then do the preprocessing and save them.
        super(EdgeText2TreeDataset, self).__init__(
            root_dir=root_dir,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_construction_name=graph_construction_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            share_vocab=share_vocab,
            pretrained_word_emb_name=pretrained_word_emb_name,
            val_split_ratio=val_split_ratio,
            word_emb_size=word_emb_size,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            nlp_processor_args=nlp_processor_args,
            enc_emb_size=enc_emb_size,
            dec_emb_size=dec_emb_size,
            min_word_vocab_freq=min_word_vocab_freq,
            max_word_vocab_size=max_word_vocab_size,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
            init_edge_vocab=init_edge_vocab,
        )
        self.data_item_type = dataitem

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'edge_vocab'."""
        return {"vocab": "vocab.pt", "data": "data.pt", "edge_vocab": "edge_vocab.pt"}
    
    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by
        each individual task-specific base class. Returns all the indices of data items
        in this file w.r.t. the whole dataset.

        For Text2TreeDataset, the format of the input file should contain lines of input,
        each line representing one record of data. The input and output is separated by
        a tab(\t).

        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )

        DataItem: input_text="list job use languageid0", output_text="job ( ANS ) ,
        language ( ANS , languageid0 )"

        Parameters
        ----------
        file_path: str
            The path of the input file.

        Returns
        -------
        list
            The indices of data items in the file w.r.t. the whole dataset.
        """
        data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                input, output = line.split("\t")
                data_item = self.data_item_type(
                    input_text=input,
                    output_text=output,
                    output_tree=None,
                    tokenizer=self.tokenizer,
                    share_vocab=self.share_vocab,
                )
                data.append(data_item)
        return data

    def vectorization(self, data_items):
        """For tree decoder we also need the vectorize the tree output."""
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]["token"]
                node_token_id = self.vocab_model.in_word_vocab.get_symbol_idx(node_token)
                graph.node_attributes[node_idx]["token_id"] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix

            token_matrix = []
            if self.init_edge_vocab:
                for edge_idx in range(graph.get_edge_num()):
                    if "token" in graph.edge_attributes[edge_idx]:
                        edge_token = graph.edge_attributes[edge_idx]["token"]
                    else:
                        edge_token = ""
                    edge_token_id = self.edge_vocab[edge_token]
                    graph.edge_attributes[edge_idx]["token_id"] = edge_token_id
                    token_matrix.append([edge_token_id])
                token_matrix = torch.tensor(token_matrix, dtype=torch.long)
                graph.edge_features["token_id"] = token_matrix
            
            if "pos_tag" in graph.graph_attributes:
                pos_vocab = [".", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
                pos_map = {pos: i for i, pos in enumerate(pos_vocab)}
                maxlen = max(len(pos_tag) for pos_tag in graph.graph_attributes["pos_tag"])
                pos_token_id = torch.zeros(len(graph.graph_attributes["pos_tag"]), maxlen, dtype=torch.long)
                for i, sentence_token in enumerate(graph.graph_attributes["pos_tag"]):
                    for j, token in enumerate(sentence_token):
                        if token in pos_map:
                            pos_token_id[i][j] = pos_map[token]
                        else:
                            print('pos_tag', token)
                graph.graph_attributes["pos_tag_id"] = pos_token_id

            if "entity_label" in graph.graph_attributes:
                entity_label = ["O", "PERSON", "LOCATION", "ORGANIZATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME", "DURATION", "SET", "EMAIL", "URL", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION", "TITLE", "IDEOLOGY", "CRIMINAL_CHARGE", "CAUSE_OF_DEATH", "HANDLE"]
                entity_map = {entity: i for i, entity in enumerate(entity_label)}
                maxlen = max(len(entity_tag) for entity_tag in graph.graph_attributes["entity_label"])
                entity_token_id = torch.zeros(len(graph.graph_attributes["entity_label"]), maxlen, dtype=torch.long)
                for i, sentence_token in enumerate(graph.graph_attributes["entity_label"]):
                    for j, token in enumerate(sentence_token):
                        if token in entity_map:
                            entity_token_id[i][j] = entity_map[token]
                        else:
                            print('entity_label', token)
                graph.graph_attributes["entity_label_id"] = entity_token_id

            if "sentence" in graph.graph_attributes:
                maxlen = max(len(sentence.strip().split()) for sentence in graph.graph_attributes["sentence"])
                seq_token_id = torch.zeros(len(graph.graph_attributes["sentence"]), maxlen, dtype=torch.long)
                for i, sentence in enumerate(graph.graph_attributes["sentence"]):
                    sentence_token = sentence.strip().split()
                    for j, token in enumerate(sentence_token):
                        seq_token_id[i][j] = self.vocab_model.in_word_vocab.get_symbol_idx(token)
                graph.graph_attributes["sentence_id"] = seq_token_id

            tgt = item.output_text
            tgt_list = self.vocab_model.out_word_vocab.get_symbol_idx_for_list(tgt.split())
            output_tree = Tree.convert_to_tree(
                tgt_list, 0, len(tgt_list), self.vocab_model.out_word_vocab
            )
            item.output_tree = output_tree

    def _vectorize_one_dataitem(cls, data_item, vocab_model, use_ie=False, edge_vocab=None):
        item = deepcopy(data_item)
        graph: GraphData = item.graph
        token_matrix = []
        for node_idx in range(graph.get_node_num()):
            node_token = graph.node_attributes[node_idx]["token"]
            node_token_id = vocab_model.in_word_vocab.get_symbol_idx(node_token)
            graph.node_attributes[node_idx]["token_id"] = node_token_id
            token_matrix.append([node_token_id])
        token_matrix = torch.tensor(token_matrix, dtype=torch.long)
        graph.node_features["token_id"] = token_matrix
        if edge_vocab is not None:
            token_matrix = []
            for edge_idx in range(graph.get_edge_num()):
                if "token" in graph.edge_attributes[edge_idx]:
                    edge_token = graph.edge_attributes[edge_idx]["token"]
                else:
                    edge_token = ""
                edge_token_id = edge_vocab[edge_token]
                graph.edge_attributes[edge_idx]["token_id"] = edge_token_id
                token_matrix.append([edge_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.edge_features["token_id"] = token_matrix

        if isinstance(item.output_text, str):
            tgt = item.output_text
            tgt_list = vocab_model.out_word_vocab.get_symbol_idx_for_list(tgt.split())
            output_tree = Tree.convert_to_tree(
                tgt_list, 0, len(tgt_list), vocab_model.out_word_vocab
            )
            item.output_tree = output_tree
        return item