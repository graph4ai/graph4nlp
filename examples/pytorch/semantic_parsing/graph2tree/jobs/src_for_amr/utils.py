from abc import abstractmethod
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from graph4nlp.pytorch.data.data import GraphData, from_batch
from graph4nlp.pytorch.data.dataset import DataItem, Text2TreeDataset

from graph4nlp.pytorch.datasets.jobs import JobsDatasetForTree
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import BertEmbedding, EmbeddingConstruction, MeanEmbedding, RNNEmbedding, WordEmbedding
from graph4nlp.pytorch.modules.utils.generic_utils import dropout_fn
from graph4nlp.pytorch.modules.utils.tree_utils import Tree

from examples.pytorch.rgcn.rgcn import RGCN
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab

warnings.filterwarnings("ignore")

class AmrEmbeddingConstruction(EmbeddingConstruction):
    """Initial graph embedding construction class.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    single_token_item : bool
        Specify whether the item (i.e., node or edge) contains single token or multiple tokens.
    emb_strategy : str
        Specify the embedding construction strategy including the following options:
            - 'w2v': use word2vec embeddings.
            - 'w2v_bilstm': use word2vec embeddings, and apply BiLSTM encoders.
            - 'w2v_bigru': use word2vec embeddings, and apply BiGRU encoders.
            - 'bert': use BERT embeddings.
            - 'bert_bilstm': use BERT embeddings, and apply BiLSTM encoders.
            - 'bert_bigru': use BERT embeddings, and apply BiGRU encoders.
            - 'w2v_bert': use word2vec and BERT embeddings.
            - 'w2v_bert_bilstm': use word2vec and BERT embeddings, and apply BiLSTM encoders.
            - 'w2v_bert_bigru': use word2vec and BERT embeddings, and apply BiGRU encoders.
        Note that if 'w2v' is not applied, `pretrained_word_emb_name` specified in Dataset APIs
        will be superseded.
    hidden_size : int, optional
        The hidden size of RNN layer, default: ``None``.
    num_rnn_layers : int, optional
        The number of RNN layers, default: ``1``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.
    fix_bert_emb : boolean, optional
        Specify whether to fix pretrained BERT embeddings, default: ``True``.
    bert_model_name : str, optional
        Specify the BERT model name, default: ``'bert-base-uncased'``.
    bert_lower_case : bool, optional
        Specify whether to lower case the input text for BERT embeddings, default: ``True``.
    word_dropout : float, optional
        Dropout ratio for word embedding, default: ``None``.
    rnn_dropout : float, optional
        Dropout ratio for RNN embedding, default: ``None``.

    Note
    ----------
        word_emb_type : str or list of str
            Specify pretrained word embedding types including "w2v", "node_edge_bert",
            or "seq_bert".
        node_edge_emb_strategy : str
            Specify node/edge embedding strategies including "mean", "bilstm" and "bigru".
        seq_info_encode_strategy : str
            Specify strategies of encoding sequential information in raw text
            data including "none", "bilstm" and "bigru". You might
            want to do this in some situations, e.g., when all the nodes are single
            tokens extracted from the raw text.

        1) single-token node (i.e., single_token_item=`True`):
            a) 'w2v', 'bert', 'w2v_bert'
            b) node_edge_emb_strategy: 'mean'
            c) seq_info_encode_strategy: 'none', 'bilstm', 'bigru'
            emb_strategy: 'w2v', 'w2v_bilstm', 'w2v_bigru',
            'bert', 'bert_bilstm', 'bert_bigru',
            'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru'

        2) multi-token node (i.e., single_token_item=`False`):
            a) 'w2v', 'bert', 'w2v_bert'
            b) node_edge_emb_strategy: 'mean', 'bilstm', 'bigru'
            c) seq_info_encode_strategy: 'none'
            emb_strategy: ('w2v', 'w2v_bilstm', 'w2v_bigru',
            'bert', 'bert_bilstm', 'bert_bigru',
            'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru')
    """

    def __init__(
        self,
        word_vocab,
        single_token_item,
        emb_strategy="w2v_bilstm",
        hidden_size=None,
        num_rnn_layers=1,
        fix_word_emb=True,
        fix_bert_emb=True,
        bert_model_name="bert-base-uncased",
        bert_lower_case=True,
        word_dropout=None,
        bert_dropout=None,
        rnn_dropout=None,
    ):
        super(EmbeddingConstruction, self).__init__()
        self.word_dropout = word_dropout
        self.bert_dropout = bert_dropout
        self.rnn_dropout = rnn_dropout
        self.single_token_item = single_token_item

        assert emb_strategy in (
            "w2v",
            "w2v_bilstm",
            "w2v_bigru",
            "bert",
            "bert_bilstm",
            "bert_bigru",
            "w2v_bert",
            "w2v_bert_bilstm",
            "w2v_bert_bigru",
            "w2v_amr",
            "w2v_bilstm_amr",
            "w2v_bilstm_amr_pos",
        ), "emb_strategy must be one of ('w2v', 'w2v_bilstm', 'w2v_bigru', 'bert', 'bert_bilstm', "
        "'bert_bigru', 'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru')"

        word_emb_type = set()
        if single_token_item:
            node_edge_emb_strategy = None
            if "w2v" in emb_strategy:
                word_emb_type.add("w2v")

            if "bert" in emb_strategy:
                word_emb_type.add("seq_bert")

            if "bilstm" in emb_strategy:
                seq_info_encode_strategy = "bilstm"
            elif "bigru" in emb_strategy:
                seq_info_encode_strategy = "bigru"
            else:
                seq_info_encode_strategy = "none"
        else:
            seq_info_encode_strategy = "none"
            if "amr" in emb_strategy:
                seq_info_encode_strategy = "bilstm"
            
            if "pos" in emb_strategy:
                word_emb_type.add("pos")
                #word_emb_type.add("entity_label")
                word_emb_type.add("position")

            if "w2v" in emb_strategy:
                word_emb_type.add("w2v")

            if "bert" in emb_strategy:
                word_emb_type.add("node_edge_bert")

            if "bilstm" in emb_strategy:
                node_edge_emb_strategy = "bilstm"
            elif "bigru" in emb_strategy:
                node_edge_emb_strategy = "bigru"
            else:
                node_edge_emb_strategy = "mean"

        word_emb_size = 0
        self.word_emb_layers = nn.ModuleDict()
        if "w2v" in word_emb_type:
            self.word_emb_layers["w2v"] = WordEmbedding(
                word_vocab.embeddings.shape[0],
                word_vocab.embeddings.shape[1],
                pretrained_word_emb=word_vocab.embeddings,
                fix_emb=fix_word_emb,
            )
            word_emb_size += word_vocab.embeddings.shape[1]

        if "node_edge_bert" in word_emb_type:
            self.word_emb_layers["node_edge_bert"] = BertEmbedding(
                name=bert_model_name, fix_emb=fix_bert_emb, lower_case=bert_lower_case
            )
            word_emb_size += self.word_emb_layers["node_edge_bert"].bert_model.config.hidden_size

        if "seq_bert" in word_emb_type:
            self.word_emb_layers["seq_bert"] = BertEmbedding(
                name=bert_model_name, fix_emb=fix_bert_emb, lower_case=bert_lower_case
            )

        if node_edge_emb_strategy in ("bilstm", "bigru"):
            self.node_edge_emb_layer = RNNEmbedding(
                word_emb_size,
                hidden_size,
                bidirectional=True,
                num_layers=num_rnn_layers,
                rnn_type="lstm" if node_edge_emb_strategy == "bilstm" else "gru",
                dropout=rnn_dropout,
            )
            rnn_input_size = hidden_size
        elif node_edge_emb_strategy == "mean":
            self.node_edge_emb_layer = MeanEmbedding()
            rnn_input_size = word_emb_size
        else:
            rnn_input_size = word_emb_size

        if "pos" in word_emb_type:
            self.word_emb_layers["pos"] = WordEmbedding(50, 50)
            rnn_input_size += 50

        if "entity_label" in word_emb_type:
            self.word_emb_layers["entity_label"] = WordEmbedding(50, 50)
            rnn_input_size += 50

        if "position" in word_emb_type:
            pass

        if "seq_bert" in word_emb_type:
            rnn_input_size += self.word_emb_layers["seq_bert"].bert_model.config.hidden_size

        if seq_info_encode_strategy in ("bilstm", "bigru"):
            self.output_size = hidden_size
            self.seq_info_encode_layer = RNNEmbedding(
                rnn_input_size,
                hidden_size,
                bidirectional=True,
                num_layers=num_rnn_layers,
                rnn_type="lstm" if seq_info_encode_strategy == "bilstm" else "gru",
                dropout=rnn_dropout,
            )

        else:
            self.output_size = rnn_input_size
            self.seq_info_encode_layer = None
        
        #self.fc = nn.Linear(376, 300)

    def forward(self, batch_gd):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        batch_gd : GraphData
            The input graph data.

        Returns
        -------
        GraphData
            The output graph data with updated node embeddings.
        """
        feat = []
        if self.single_token_item:  # single-token node graph
            token_ids = batch_gd.batch_node_features["token_id"]
            if "w2v" in self.word_emb_layers:
                word_feat = self.word_emb_layers["w2v"](token_ids).squeeze(-2)
                word_feat = dropout_fn(
                    word_feat, self.word_dropout, shared_axes=[-2], training=self.training
                )
                feat.append(word_feat)

        else:  # multi-token node graph
            token_ids = batch_gd.node_features["token_id"]
            if "w2v" in self.word_emb_layers:
                word_feat = self.word_emb_layers["w2v"](token_ids)
                word_feat = dropout_fn(
                    word_feat, self.word_dropout, shared_axes=[-2], training=self.training
                )
                feat.append(word_feat)
            if any(batch_gd.batch_graph_attributes):
                tot = 0
                gd_list = from_batch(batch_gd)
                for i, g in enumerate(gd_list):
                    sentence_id = g.graph_attributes["sentence_id"].to(batch_gd.device)
                    seq_feat = []
                    if "w2v" in self.word_emb_layers:
                        word_feat = self.word_emb_layers["w2v"](sentence_id)
                        word_feat = dropout_fn(
                            word_feat, self.word_dropout, shared_axes=[-2], training=self.training
                        )
                        seq_feat.append(word_feat)
                    else:
                        RuntimeError("No word embedding layer")
                    if "pos" in self.word_emb_layers:
                        sentence_pos = g.graph_attributes["pos_tag_id"].to(batch_gd.device)
                        pos_feat = self.word_emb_layers["pos"](sentence_pos)
                        pos_feat = dropout_fn(
                            pos_feat, self.word_dropout, shared_axes=[-2], training=self.training
                        )
                        seq_feat.append(pos_feat)
                    
                    if "entity_label" in self.word_emb_layers:
                        sentence_entity_label = g.graph_attributes["entity_label_id"].to(batch_gd.device)
                        entity_label_feat = self.word_emb_layers["entity_label"](sentence_entity_label)
                        entity_label_feat = dropout_fn(
                            entity_label_feat, self.word_dropout, shared_axes=[-2], training=self.training
                        )
                        seq_feat.append(entity_label_feat)

                    seq_feat = torch.cat(seq_feat, dim=-1)
                    
                    raw_tokens = [dd.strip().split() for dd in g.graph_attributes["sentence"]]
                    l = [len(s) for s in raw_tokens]
                    rnn_state = self.seq_info_encode_layer(
                        seq_feat, torch.LongTensor(l).to(batch_gd.device)
                    )
                    if isinstance(rnn_state, (tuple, list)):
                        rnn_state = rnn_state[0]

                    # update node features
                    for j in range(g.get_node_num()):
                        id = g.node_attributes[j]["sentence_id"]
                        if g.node_attributes[j]["id"] in batch_gd.batch_graph_attributes[i]["mapping"][id]:
                            rel_list = batch_gd.batch_graph_attributes[i]["mapping"][id][g.node_attributes[j]["id"]]
                            state = []
                            for rel in rel_list:
                                if rel[1] == "node":
                                    state.append(rnn_state[id][rel[0]])
                            # replace embedding of the node
                            if len(state) > 0:
                                feat[0][tot + j][0] = torch.stack(state, 0).mean(0)

                    tot += g.get_node_num()

            if "node_edge_bert" in self.word_emb_layers:
                input_data = [
                    batch_gd.node_attributes[i]["token"].strip().split(" ")
                    for i in range(batch_gd.get_node_num())
                ]
                node_edge_bert_feat = self.word_emb_layers["node_edge_bert"](input_data)
                node_edge_bert_feat = dropout_fn(
                    node_edge_bert_feat, self.bert_dropout, shared_axes=[-2], training=self.training
                )
                feat.append(node_edge_bert_feat)

            if len(feat) > 0:
                feat = torch.cat(feat, dim=-1)
                if not any(batch_gd.batch_graph_attributes):
                    node_token_lens = torch.clamp((token_ids != Vocab.PAD).sum(-1), min=1)
                    feat = self.node_edge_emb_layer(feat, node_token_lens)
                else:
                    feat = feat.squeeze(dim=1)
                if isinstance(feat, (tuple, list)):
                    feat = feat[-1]

                feat = batch_gd.split_features(feat)

        if (self.seq_info_encode_layer is None and "seq_bert" not in self.word_emb_layers) or any(batch_gd.batch_graph_attributes):
            if isinstance(feat, list):
                feat = torch.cat(feat, -1)

            batch_gd.batch_node_features["node_feat"] = feat

            return batch_gd
        else:  # single-token node graph
            new_feat = feat
            if "seq_bert" in self.word_emb_layers:
                gd_list = from_batch(batch_gd)
                raw_tokens = [
                    [gd.node_attributes[i]["token"] for i in range(gd.get_node_num())]
                    for gd in gd_list
                ]
                bert_feat = self.word_emb_layers["seq_bert"](raw_tokens)
                bert_feat = dropout_fn(
                    bert_feat, self.bert_dropout, shared_axes=[-2], training=self.training
                )
                new_feat.append(bert_feat)

            new_feat = torch.cat(new_feat, -1)
            if self.seq_info_encode_layer is None:
                batch_gd.batch_node_features["node_feat"] = new_feat

                return batch_gd

            rnn_state = self.seq_info_encode_layer(
                new_feat, torch.LongTensor(batch_gd._batch_num_nodes).to(batch_gd.device)
            )
            if isinstance(rnn_state, (tuple, list)):
                rnn_state = rnn_state[0]

            batch_gd.batch_node_features["node_feat"] = rnn_state

            return batch_gd

class AMRGraphEmbeddingInitialization(nn.Module):
    def __init__(
        self,
        word_vocab,
        embedding_style,
        hidden_size=None,
        fix_word_emb=True,
        fix_bert_emb=True,
        word_dropout=None,
        rnn_dropout=None,
    ):
        super(AMRGraphEmbeddingInitialization, self).__init__()
        self.embedding_layer = AmrEmbeddingConstruction(
            word_vocab,
            embedding_style["single_token_item"],
            emb_strategy=embedding_style["emb_strategy"],
            hidden_size=hidden_size,
            num_rnn_layers=embedding_style.get("num_rnn_layers", 1),
            fix_word_emb=fix_word_emb,
            fix_bert_emb=fix_bert_emb,
            bert_model_name=embedding_style.get("bert_model_name", "bert-base-uncased"),
            bert_lower_case=embedding_style.get("bert_lower_case", True),
            word_dropout=word_dropout,
            rnn_dropout=rnn_dropout,
        )

    @abstractmethod
    def forward(self, graph_data: GraphData):
        return self.embedding_layer(graph_data)
class AMRDataItem(DataItem):
    def __init__(self, input_text, output_text, tokenizer, output_tree=None, share_vocab=True):
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
    
    def extract_edge_tokens(self):
        g: GraphData = self.graph
        edge_tokens = []
        for i in range(g.get_edge_num()):
            edge_tokens.append(g.edge_attributes[i]["token"])
        return edge_tokens

class RGCNGraph2Tree(Graph2Tree):
    def __init__(
        self,
        vocab_model,
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
                num_rels=80,
                num_bases=4,
                gpu=0,
            )
        else:
            raise NotImplementedError()
    
    @classmethod
    def from_args(cls, opt, vocab_model):
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
        return cls(vocab_model=vocab_model,  **args)

class AMRGraph2Tree(RGCNGraph2Tree):
    def __init__(
        self,
        vocab_model,
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
        style = embedding_style["emb_strategy"]
        embedding_style["emb_strategy"] = "w2v_bilstm"
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
        embedding_style["emb_strategy"] = style
        self.graph_initializer = AMRGraphEmbeddingInitialization(
            word_vocab=vocab_model.in_word_vocab,
            embedding_style=embedding_style,
            hidden_size=emb_hidden_size,
            word_dropout=emb_word_dropout,
            rnn_dropout=emb_rnn_dropout,
            fix_word_emb=emb_fix_word_emb,
            fix_bert_emb=emb_fix_bert_emb,
        )
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
        is_hetero=True,
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
            is_hetero=True,
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
                    edge_token_id = self.vocab_model.edge_vocab[edge_token]
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

    def _vectorize_one_dataitem(cls, data_item, vocab_model, use_ie=False):
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
        if hasattr(vocab_model, "edge_vocab"):
            token_matrix = []
            for edge_idx in range(graph.get_edge_num()):
                if "token" in graph.edge_attributes[edge_idx]:
                    edge_token = graph.edge_attributes[edge_idx]["token"]
                else:
                    edge_token = ""
                edge_token_id = vocab_model.edge_vocab[edge_token]
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


class EdgeText2TreeDataset(JobsDatasetForTree):
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
        is_hetero=True,
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
            is_hetero=is_hetero,
        )
        self.data_item_type = dataitem

    @property
    def processed_file_names(self):
        """At least 2 reserved keys should be fiiled: 'vocab', 'data'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}
    
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
            if self.is_hetero:
                for edge_idx in range(graph.get_edge_num()):
                    if "token" in graph.edge_attributes[edge_idx]:
                        edge_token = graph.edge_attributes[edge_idx]["token"]
                    else:
                        edge_token = ""
                    edge_token_id = self.vocab_model.edge_vocab[edge_token]
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

    def _vectorize_one_dataitem(cls, data_item, vocab_model, use_ie=False):
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
        # test if vocab_model has edge_vocab attribute
        if hasattr(vocab_model, "edge_vocab"):
            token_matrix = []
            for edge_idx in range(graph.get_edge_num()):
                if "token" in graph.edge_attributes[edge_idx]:
                    edge_token = graph.edge_attributes[edge_idx]["token"]
                else:
                    edge_token = ""
                edge_token_id = vocab_model.edge_vocab[edge_token]
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