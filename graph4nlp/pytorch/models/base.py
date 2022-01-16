import abc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import (  # noqa
    NodeEmbeddingBasedRefinedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_embedding_initialization.graph_embedding_initialization import (  # noqa
    GraphEmbeddingInitialization,
)
from graph4nlp.pytorch.modules.graph_embedding_learning.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding_learning.gcn import GCN
from graph4nlp.pytorch.modules.graph_embedding_learning.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding_learning.graphsage import GraphSAGE


class Graph2XBase(nn.Module):
    def __init__(
        self,
        vocab_model,
        embedding_style,
        graph_name,
        emb_input_size,
        emb_hidden_size,
        gnn,
        gnn_num_layers,
        gnn_direction_option,
        gnn_input_size,
        gnn_hidden_size,
        gnn_output_size,
        # dropout
        emb_word_dropout=0.0,
        emb_rnn_dropout=0.0,
        emb_fix_word_emb=False,
        emb_fix_bert_emb=False,
        gnn_feats_dropout=0.0,
        gnn_attn_dropout=0.0,
        **kwargs
    ):
        super(Graph2XBase, self).__init__()
        self.vocab_model = vocab_model
        self._build_embedding_encoder(
            graph_name=graph_name,
            embedding_style=embedding_style,
            vocab_model=vocab_model,
            emb_input_size=emb_input_size,
            emb_hidden_size=emb_hidden_size,
            emb_word_dropout=emb_word_dropout,
            emb_rnn_dropout=emb_rnn_dropout,
            emb_fix_word_emb=emb_fix_word_emb,
            emb_fix_bert_emb=emb_fix_bert_emb,
            **kwargs
        )

        self._build_gnn_encoder(
            gnn=gnn,
            num_layers=gnn_num_layers,
            input_size=gnn_input_size,
            hidden_size=gnn_hidden_size,
            output_size=gnn_output_size,
            direction_option=gnn_direction_option,
            feats_dropout=gnn_feats_dropout,
            gnn_attn_dropout=gnn_attn_dropout,
            **kwargs
        )

    def _build_embedding_encoder(
        self,
        graph_name,
        embedding_style,
        vocab_model,
        emb_input_size,
        emb_hidden_size,
        emb_rnn_dropout,
        emb_word_dropout,
        # dynamic parameters
        emb_sim_metric_type=None,
        emb_num_heads=None,
        emb_top_k_neigh=None,
        emb_epsilon_neigh=None,
        emb_smoothness_ratio=None,
        emb_connectivity_ratio=None,
        emb_sparsity_ratio=None,
        emb_alpha_fusion=None,
        emb_fix_word_emb=False,
        emb_fix_bert_emb=False,
        **kwargs
    ):

        if not isinstance(graph_name, str):
            raise ValueError("graph_name parameter should be str")

        self.graph_initializer = GraphEmbeddingInitialization(
            word_vocab=vocab_model.in_word_vocab,
            embedding_style=embedding_style,
            hidden_size=emb_hidden_size,
            word_dropout=emb_word_dropout,
            rnn_dropout=emb_rnn_dropout,
            fix_word_emb=emb_fix_word_emb,
            fix_bert_emb=emb_fix_bert_emb,
        )

        if graph_name == "node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                sim_metric_type=emb_sim_metric_type,
                num_heads=emb_num_heads,
                top_k_neigh=emb_top_k_neigh,
                epsilon_neigh=emb_epsilon_neigh,
                smoothness_ratio=emb_smoothness_ratio,
                connectivity_ratio=emb_connectivity_ratio,
                sparsity_ratio=emb_sparsity_ratio,
                input_size=emb_input_size,
                hidden_size=emb_hidden_size,
            )
        elif graph_name == "node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                emb_alpha_fusion,
                sim_metric_type=emb_sim_metric_type,
                num_heads=emb_num_heads,
                top_k_neigh=emb_top_k_neigh,
                epsilon_neigh=emb_epsilon_neigh,
                smoothness_ratio=emb_smoothness_ratio,
                connectivity_ratio=emb_connectivity_ratio,
                sparsity_ratio=emb_sparsity_ratio,
                input_size=emb_input_size,
                hidden_size=emb_hidden_size,
            )

        self.enc_word_emb = (
            self.graph_initializer.embedding_layer.word_emb_layers["w2v"]
            if "w2v" in self.graph_initializer.embedding_layer.word_emb_layers
            else None
        )
        self.graph_name = graph_name

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
        if gnn == "gat":
            self.gnn_encoder = GAT(
                num_layers,
                input_size,
                hidden_size,
                output_size,
                gnn_heads,
                direction_option=direction_option,
                feat_drop=feats_dropout,
                attn_drop=gnn_attn_dropout,
                activation=gnn_activation,
                residual=gnn_residual,
                allow_zero_in_degree=gnn_allow_zero_in_degree,
            )
        elif gnn == "ggnn":
            self.gnn_encoder = GGNN(
                num_layers,
                input_size,
                hidden_size,
                output_size,
                direction_option=direction_option,
                use_edge_weight=gnn_use_edge_weight,
                feat_drop=feats_dropout,
                n_etypes=gnn_n_etypes,
            )
        elif gnn == "graphsage":
            self.gnn_encoder = GraphSAGE(
                num_layers,
                input_size,
                hidden_size,
                output_size,
                aggregator_type=gnn_aggregator_type,
                direction_option=direction_option,
                feat_drop=feats_dropout,
                activation=gnn_activation,
                bias=gnn_bias,
                use_edge_weight=gnn_use_edge_weight,
            )
        elif gnn == "gcn":
            self.gnn_encoder = GCN(
                num_layers,
                input_size,
                hidden_size,
                output_size,
                direction_option=direction_option,
                weight=gnn_weight,
                gcn_norm=gnn_gcn_norm,
                allow_zero_in_degree=gnn_allow_zero_in_degree,
                activation=gnn_activation,
                use_edge_weight=gnn_use_edge_weight,
            )
        else:
            raise NotImplementedError()

    @abc.abstractmethod
    def __build_decoder(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, raw_sentences):
        raise NotImplementedError()

    def save_checkpoint(self, save_path, checkpoint_name):
        """
            The API for saving the model.
        Parameters
        ----------
        save_path : str
            The root path.
        checkpoint_name : str
            The name of the checkpoint.
        Returns
        -------

        """
        checkpoint_path = os.path.join(save_path, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self, checkpoint_path)

    @classmethod
    def load_checkpoint(cls, load_path, checkpoint_name):
        """
            The API to load the model.

        Parameters
        ----------
        load_path : str
            The root path to load the model.
        checkpoint_name : str
            The name of the model to be loaded.

        Returns
        -------
        Graph2XBase
        """
        checkpoint_path = os.path.join(load_path, checkpoint_name)
        model = torch.load(checkpoint_path)
        return model
