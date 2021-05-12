import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder
from graph4nlp.pytorch.modules.utils.tree_utils import Vocab

from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.graph_embedding import *


class Graph2Tree(nn.Module):
    def __init__(self, src_vocab,
                 tgt_vocab,
                 use_copy,
                 gnn_type,
                 direction_option,
                 graph_construction_type,
                 enc_hidden_size,
                 dec_hidden_size,
                 dropout_for_word_embedding,
                 dropout_for_encoder_feature,
                 dropout_for_encoder_attn,
                 dropout_for_decoder,
                 device,
                 criterion,
                 teacher_force_ratio,
                 max_dec_seq_length,
                 max_dec_tree_depth,
                 embedding_style,
                 K,
                 gat_head,
                 sage_aggr,
                 attn_type,
                 use_sibling,
                 use_share_vocab):
        super(Graph2Tree, self).__init__()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.use_copy = use_copy
        self.input_size = self.src_vocab.vocab_size
        self.output_size = self.tgt_vocab.vocab_size
        self.criterion = criterion
        self.use_share_vocab = use_share_vocab

        if graph_construction_type == "DependencyGraph":
            self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=self.src_vocab,
                                                                   hidden_size=enc_hidden_size, 
                                                                   word_dropout=dropout_for_word_embedding, 
                                                                   rnn_dropout=dropout_for_word_embedding, 
                                                                   fix_word_emb=False)
            self.use_edge_weight = False
        elif graph_construction_type == "ConstituencyGraph":
            self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                     vocab=self.src_vocab,
                                                                     hidden_size=enc_hidden_size,
                                                                     word_dropout=dropout_for_word_embedding,
                                                                     rnn_dropout=dropout_for_word_embedding,
                                                                     fix_word_emb=False)
            self.use_edge_weight = False
        elif graph_construction_type == "DynamicGraph_node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(self.src_vocab,
                                                                      embedding_style,
                                                                      sim_metric_type='weighted_cosine',
                                                                      num_heads=1,
                                                                      top_k_neigh=None,
                                                                      epsilon_neigh=0.5,
                                                                      smoothness_ratio=0.1,
                                                                      connectivity_ratio=0.05,
                                                                      sparsity_ratio=0.1,
                                                                      input_size=enc_hidden_size,
                                                                      hidden_size=enc_hidden_size,
                                                                      fix_word_emb=False,
                                                                      word_dropout=dropout_for_word_embedding,
                                                                      rnn_dropout=dropout_for_word_embedding
                                                                      )
            self.use_edge_weight = True
        elif graph_construction_type == "DynamicGraph_node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(self.src_vocab,
                                                                             embedding_style,
                                                                             0.2,
                                                                             sim_metric_type="weighted_cosine",
                                                                             num_heads=1,
                                                                             top_k_neigh=None,
                                                                             epsilon_neigh=0.5,
                                                                             smoothness_ratio=0.1,
                                                                             connectivity_ratio=0.05,
                                                                             sparsity_ratio=0.1,
                                                                             input_size=enc_hidden_size,
                                                                             hidden_size=enc_hidden_size,
                                                                             fix_word_emb=False,
                                                                             word_dropout=dropout_for_word_embedding,
                                                                             rnn_dropout=dropout_for_word_embedding,
                                                                             device=device)
            self.use_edge_weight = True
        else:
            raise NotImplementedError()

        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
        if self.use_share_vocab == 0:
            self.tgt_word_embedding = nn.Embedding(self.tgt_vocab.vocab_size, dec_hidden_size, 
                                                padding_idx=self.tgt_vocab.get_symbol_idx(self.tgt_vocab.pad_token),
                                                _weight=torch.from_numpy(self.tgt_vocab.embeddings).float())

        if gnn_type == "GAT":
            self.encoder = GAT(K, enc_hidden_size, enc_hidden_size, enc_hidden_size, gat_head,
                               direction_option=direction_option, feat_drop=dropout_for_encoder_feature,
                               attn_drop=dropout_for_encoder_attn, activation=F.relu, residual=True)
        elif gnn_type == "GGNN":
            self.encoder = GGNN(K, enc_hidden_size, enc_hidden_size, enc_hidden_size,
                                feat_drop=dropout_for_encoder_feature, use_edge_weight=self.use_edge_weight,
                                direction_option=direction_option)
        elif gnn_type == "SAGE":
            self.encoder = GraphSAGE(K, enc_hidden_size, enc_hidden_size, enc_hidden_size,
                                     sage_aggr, direction_option=direction_option, feat_drop=dropout_for_encoder_feature,
                                     activation=F.relu, bias=True, use_edge_weight=self.use_edge_weight)
        elif gnn_type == "GCN":
            self.encoder = GCN(K, enc_hidden_size, enc_hidden_size, enc_hidden_size,
                               direction_option=direction_option, gcn_norm="both", activation=F.relu,
                               use_edge_weight=self.use_edge_weight)
        else:
            print("Wrong gnn type, please use GCN GAT GGNN or SAGE")
            raise NotImplementedError()

        self.decoder = StdTreeDecoder(attn_type=attn_type,
                                      embeddings=self.word_emb if self.use_share_vocab else self.tgt_word_embedding,
                                      enc_hidden_size=enc_hidden_size,
                                      dec_emb_size=self.tgt_vocab.embedding_dims,
                                      dec_hidden_size=dec_hidden_size,
                                      output_size=self.output_size,
                                      device=device,
                                      criterion=self.criterion,
                                      teacher_force_ratio=teacher_force_ratio,
                                      use_sibling=use_sibling,
                                      use_copy=self.use_copy,
                                      dropout_for_decoder=dropout_for_decoder,
                                      max_dec_seq_length=max_dec_seq_length,
                                      max_dec_tree_depth=max_dec_tree_depth,
                                      tgt_vocab=self.tgt_vocab)

    def forward(self, batch_graph, tgt_tree_batch, oov_dict=None):
        batch_graph = self.graph_topology(batch_graph)
        batch_graph = self.encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

        loss = self.decoder(g=batch_graph, tgt_tree_batch=tgt_tree_batch, oov_dict=oov_dict)
        return loss

    def init(self, init_weight):
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        for name, param in self.named_parameters():
            if param.requires_grad:
                if ("word_embedding" in name) or ("word_emb_layer" in name) or ("bert_embedding" in name):
                    pass
                else:
                    if len(param.size()) >= 2:
                        if "rnn" in name:
                            init.orthogonal_(param)
                        else:
                            init.xavier_uniform_(param, gain=1.0)
                    else:
                        init.uniform_(param, -init_weight, init_weight)
    @classmethod
    def from_args(cls, opt, src_vocab, tgt_vocab, device, embedding_style, criterion):
        return cls(src_vocab=src_vocab, tgt_vocab=tgt_vocab, use_copy=opt.use_copy, gnn_type=opt.gnn_type, 
                    direction_option=opt.direction_option, graph_construction_type=opt.graph_construction_type,
                    enc_hidden_size=opt.enc_hidden_size, dec_hidden_size=opt.dec_hidden_size, 
                    dropout_for_word_embedding=opt.dropout_for_word_embedding, dropout_for_encoder_feature=opt.dropout_for_encoder,
                    dropout_for_encoder_attn=opt.dropout_for_encoder, dropout_for_decoder=opt.dropout_for_decoder, 
                    device=device, criterion=criterion, teacher_force_ratio=opt.teacher_force_ratio, max_dec_seq_length=opt.max_dec_seq_length, 
                    max_dec_tree_depth=opt.max_dec_tree_depth, embedding_style=embedding_style, K=opt.K, 
                    gat_head=[int(i) for i in opt.gat_head.split(',')], sage_aggr=opt.sage_aggr, 
                    attn_type=opt.attn_type, use_sibling=opt.use_sibling, use_share_vocab=opt.use_share_vocab)