import torch.nn as nn
import abc

import torch.nn as nn
import torch.nn.functional as F

from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import \
    ConstituencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import \
    NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import \
    NodeEmbeddingBasedRefinedGraphConstruction
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.gcn import GCN
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE


class Graph2XBase(nn.Module):
    def __init__(self, vocab_model, encoder_hidden_size,
                 graph_type, direction_option,
                 gnn, gnn_layer_number, embedding_style,
                 device, feats_dropout=0.2, attn_dropout=0.2, rnn_dropout=0.2, fix_word_emb=False,
                 **kwargs):
        super(Graph2XBase, self).__init__()

        self._build_embedding_encoder(graph_type=graph_type, embedding_style=embedding_style, vocab_model=vocab_model,
                                      hidden_size=encoder_hidden_size, word_dropout=feats_dropout,
                                      rnn_dropout=rnn_dropout, device=device, fix_word_emb=fix_word_emb, **kwargs)

        self._build_gnn_encoder(gnn=gnn, layer_number=gnn_layer_number, hidden_size=encoder_hidden_size,
                                direction_option=direction_option, feats_dropout=feats_dropout,
                                attn_dropout=attn_dropout, **kwargs)

    def _build_embedding_encoder(self, graph_type, embedding_style, vocab_model, hidden_size, rnn_dropout, word_dropout,
                                 device,
                                 # dynamic parameters
                                 sim_metric_type=None, num_heads=None, top_k_neigh=None, epsilon_neigh=None,
                                 smoothness_ratio=None, connectivity_ratio=None, sparsity_ratio=None, alpha_fusion=None,
                                 fix_word_emb=False, **kwargs):

        self.use_edge_weight = False

        if not isinstance(graph_type, str):
            raise ValueError("graph_type parameter should be str")

        if graph_type == "dependency":
            self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab_model.in_word_vocab,
                                                                   hidden_size=hidden_size,
                                                                   word_dropout=word_dropout,
                                                                   rnn_dropout=rnn_dropout, device=device,
                                                                   fix_word_emb=fix_word_emb)
        elif graph_type == "constituency":
            self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                     vocab=vocab_model.in_word_vocab,
                                                                     hidden_size=hidden_size, device=device,
                                                                     word_dropout=word_dropout,
                                                                     rnn_dropout=rnn_dropout,
                                                                     fix_word_emb=fix_word_emb)
        elif graph_type == "node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                vocab_model.in_word_vocab,
                embedding_style,
                sim_metric_type=sim_metric_type,
                num_heads=num_heads,
                top_k_neigh=top_k_neigh,
                epsilon_neigh=epsilon_neigh,
                smoothness_ratio=smoothness_ratio,
                connectivity_ratio=connectivity_ratio,
                sparsity_ratio=sparsity_ratio,
                input_size=hidden_size,
                hidden_size=hidden_size,
                fix_word_emb=fix_word_emb,
                word_dropout=word_dropout,
                rnn_dropout=rnn_dropout,
                device=device)
            self.use_edge_weight = True
        elif graph_type == "node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                vocab_model.in_word_vocab,
                embedding_style,
                alpha_fusion,
                sim_metric_type=sim_metric_type,
                num_heads=num_heads,
                top_k_neigh=top_k_neigh,
                epsilon_neigh=epsilon_neigh,
                smoothness_ratio=smoothness_ratio,
                connectivity_ratio=connectivity_ratio,
                sparsity_ratio=sparsity_ratio,
                input_size=hidden_size,
                hidden_size=hidden_size,
                fix_word_emb=fix_word_emb,
                word_dropout=word_dropout,
                rnn_dropout=rnn_dropout,
                device=device)
            self.use_edge_weight = True
        else:
            raise NotImplementedError()
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer

    def _build_gnn_encoder(self, gnn, layer_number, hidden_size, direction_option, feats_dropout, attn_dropout,
                           activation=F.relu,
                           gat_heads=None, gat_use_residual=True,
                           graphsage_aggregator_type="lstm", **kwargs):

        if gnn == "GAT":
            self.gnn_encoder = GAT(layer_number, hidden_size, hidden_size, hidden_size, gat_heads,
                                   direction_option=direction_option,
                                   feat_drop=feats_dropout, attn_drop=attn_dropout, activation=activation,
                                   residual=gat_use_residual)
        elif gnn == "GGNN":
            self.gnn_encoder = GGNN(layer_number, hidden_size, hidden_size, direction_option=direction_option,
                                    use_edge_weight=self.use_edge_weight, dropout=feats_dropout)
        elif gnn == "GraphSage":
            self.gnn_encoder = GraphSAGE(layer_number, hidden_size, hidden_size, hidden_size,
                                         aggregator_type=graphsage_aggregator_type,
                                         direction_option=direction_option, feat_drop=feats_dropout,
                                         activation=activation, bias=True, use_edge_weight=self.use_edge_weight)
        elif gnn == "GCN":
            self.gnn_encoder = GCN(layer_number, hidden_size, hidden_size, hidden_size,
                                   direction_option=direction_option,
                                   allow_zero_in_degree=True, activation=activation)
        else:
            raise NotImplementedError()

    @abc.abstractmethod
    def __build_decoder(self):
        raise NotImplementedError()
