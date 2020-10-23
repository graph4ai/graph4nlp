from graph4nlp.pytorch.data.data import from_batch, GraphData
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import WordEmbedding, RNNEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import Graph2seqLoss
import onmt
from .onmt_decoder import InputFeedRNNDecoder


class Graph2seq(nn.Module):
    def __init__(self, vocab, gnn, device, word_emb_size=300, rnn_dropout=0.2, word_dropout=0.2, hidden_size=300,
                 direction_option='undirected'):
        super(Graph2seq, self).__init__()

        self.vocab = vocab

        self.embeddings = onmt.modules.Embeddings(word_emb_size, self.vocab.in_word_vocab.embeddings.shape[0],
                                             word_padding_idx=vocab.in_word_vocab.PAD)

        self.seq_encoder = onmt.encoders.RNNEncoder(hidden_size=hidden_size, num_layers=1,
                                   rnn_type="LSTM", bidirectional=True,
                                   embeddings=self.embeddings)

        self.seq_decoder = InputFeedRNNDecoder(
            hidden_size=hidden_size, num_layers=1, bidirectional_encoder=True, context_gate="both",
            dropout=0.3,
            rnn_type="LSTM", embeddings=self.embeddings)

        self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)
        self.out_project = nn.Linear(hidden_size, self.vocab.in_word_vocab.embeddings.shape[0], bias=False)
        self.device = device

    def forward(self, src_seq, src_len, tgt_seq=None, require_loss=True):
        src_seq = src_seq.transpose(0, 1)
        if tgt_seq is not None:
            tgt_seq = tgt_seq.transpose(0, 1)
        enc_state, memory_bank, lengths = self.seq_encoder(src_seq.unsqueeze(-1), src_len)

        # onmt decoder
        self.seq_decoder.init_state(src_seq.unsqueeze(-1), memory_bank, enc_state)


        if tgt_seq is not None:
            dec_outs, attns = self.seq_decoder(tgt=tgt_seq.unsqueeze(-1), memory_bank=memory_bank, memory_lengths=lengths, with_align=False)
        else:
            dec_outs, attns = self.seq_decoder(memory_bank=memory_bank,
                                               memory_lengths=lengths, with_align=False)
        prob = torch.softmax(dec_outs, dim=-1)


        if require_loss:
            loss = self.loss_calc(prob, tgt_seq.transpose(0, 1))
            return prob, loss
        else:
            return prob