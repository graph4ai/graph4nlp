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
    def __init__(self, vocab, gnn, device, word_emb_size=300, rnn_dropout=0.2, word_dropout=0.2, hidden_size=300,
                 direction_option='undirected'):
        super(Graph2seq, self).__init__()

        self.vocab = vocab
        embedding_style = {'single_token_item': True,
                           'emb_strategy': "w2v_bilstm",
                           'num_rnn_layers': 1}
        self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=hidden_size,
                                                               rnn_dropout=rnn_dropout, word_dropout=word_dropout,
                                                               device=device,
                                                               fix_word_emb=False)
        if gnn == "GAT":
            self.gnn_encoder = GAT(3, hidden_size, hidden_size, hidden_size, [2, 2, 1], direction_option=direction_option,
                                   feat_drop=0.2, attn_drop=0.2, activation=F.relu, residual=True)
        elif gnn == "GGNN":
            self.gnn_encoder = GGNN(3, hidden_size, hidden_size, direction_option=direction_option, dropout=0.2)
        elif gnn == "Graphsage":
            self.gnn_encoder = GraphSAGE(2, hidden_size, hidden_size, hidden_size, aggregator_type="lstm",
                                         direction_option=direction_option, feat_drop=0.4)
        else:
            raise NotImplementedError("Please define your graph embedding method: {}".format(gnn))

        self.word_emb = nn.Embedding(len(self.vocab.out_word_vocab), word_emb_size).from_pretrained(
            torch.from_numpy(self.vocab.out_word_vocab.embeddings).float(), freeze=False)

        self.seq_decoder = StdRNNDecoder(max_decoder_step=200,
                                         decoder_input_size=2*hidden_size if direction_option == 'bi_sep' else hidden_size,
                                         decoder_hidden_size=hidden_size, graph_pooling_strategy=None,
                                         word_emb=self.word_emb, vocab=self.vocab.out_word_vocab,
                                         attention_type="sep_diff_encoder_type", fuse_strategy="concatenate",
                                         rnn_emb_input_size=hidden_size, use_coverage=False,
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
