from graph4nlp.pytorch.data.data import from_batch, GraphData
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import Graph2seqLoss


class Graph2seq(nn.Module):
    def __init__(self, vocab, device, hidden_size=512, direction_option='undirected'):
        super(Graph2seq, self).__init__()

        self.vocab = vocab
        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"}
        self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=hidden_size, dropout=0.2, device=device,
                                                               fix_word_emb=False)
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer
        self.gnn_encoder = GAT(3, hidden_size, hidden_size, hidden_size, [2, 2, 1], direction_option=direction_option,
                               feat_drop=0.2, attn_drop=0.2, activation=F.relu, residual=True)
        # self.gnn_encoder = GGNN(3, hidden_size, hidden_size, direction_option=direction_option)
        # self.gnn_encoder = GraphSAGE(3, hidden_size, hidden_size, hidden_size, aggregator_type="lstm",
        #                              direction_option=direction_option, feat_drop=0.4)
        self.seq_decoder = StdRNNDecoder(max_decoder_step=200,
                                         decoder_input_size=2*hidden_size if direction_option == 'bi_sep' else hidden_size,
                                         decoder_hidden_size=hidden_size, graph_pooling_strategy=None,
                                         word_emb=self.word_emb, vocab=self.vocab.out_word_vocab,
                                         attention_type="sep_diff_encoder_type", fuse_strategy="concatenate",
                                         rnn_emb_input_size=hidden_size, use_coverage=True,
                                         tgt_emb_as_output_layer=False, device=self.graph_topology.device,
                                         dropout=0.3)
        self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)
        # self.loss_cover = CoverageLoss(0.3)

    def forward(self, graph_list, tgt=None, require_loss=True):
        batch_graph = self.graph_topology(graph_list)

        # run GNN
        batch_graph: GraphData = self.gnn_encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

        # down-task
        prob, enc_attn_weights, coverage_vectors = self.seq_decoder(from_batch(batch_graph), tgt_seq=tgt)
        if require_loss:
            loss = self.loss_calc(prob, tgt)
            # cover_loss = self.loss_cover(prob.shape[0], enc_attn_weights, coverage_vectors)
            return prob, loss
        else:
            return prob
