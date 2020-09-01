from graph4nlp.pytorch.data.data import from_batch, GraphData
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import NodeEmbeddingBasedRefinedGraphConstruction

from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from .loss import *
import torch.nn.functional as F


class Graph2seq(nn.Module):
    def __init__(self, vocab, hidden_size=300, graph_type="dependency", direction_option='undirected',
                 gnn="GAT",
                 device=None, emb_dropout=0.2, feats_dropout=0.2, rnn_dropout=0.3):
        super(Graph2seq, self).__init__()

        self.vocab = vocab
        self.graph_type = graph_type
        use_edge_weight = False
        embedding_style = {'word_emb_type': 'w2v',
                           'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"
                           }
        if self.graph_type == "dependency":
            self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=hidden_size, dropout=emb_dropout, device=device,
                                                                   fix_word_emb=False)
        elif self.graph_type == "constituency":
            self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                     vocab=vocab.in_word_vocab,
                                                                     hidden_size=hidden_size, dropout=emb_dropout, device=device,
                                                                     fix_word_emb=False)
        elif self.graph_type == "node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                vocab.in_word_vocab,
                embedding_style,
                sim_metric_type='weighted_cosine',
                num_heads=1,
                top_k_neigh=None,
                epsilon_neigh=0.5,
                smoothness_ratio=0.1,
                connectivity_ratio=0.05,
                sparsity_ratio=0.1,
                input_size=hidden_size,
                hidden_size=hidden_size,
                fix_word_emb=False,
                word_dropout=emb_dropout,
                dropout=rnn_dropout,
                device=device)
            use_edge_weight = True
        elif self.graph_type == "node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                vocab.in_word_vocab,
                embedding_style,
                0.2,
                sim_metric_type="weighted_cosine",
                num_heads=1,
                top_k_neigh=None,
                epsilon_neigh=0.5,
                smoothness_ratio=0.1,
                connectivity_ratio=0.05,
                sparsity_ratio=0.1,
                input_size=hidden_size,
                hidden_size=hidden_size,
                fix_word_emb=False,
                word_dropout=emb_dropout,
                dropout=rnn_dropout,
                device=device)
            use_edge_weight = True
        else:
            raise NotImplementedError()
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer

        if gnn == "GAT":
            self.gnn_encoder = GAT(3, hidden_size, hidden_size, hidden_size, [2, 2, 1],
                                   direction_option=direction_option,
                                   feat_drop=feats_dropout, attn_drop=feats_dropout, activation=F.relu, residual=True)
        elif gnn == "GGNN":
            self.gnn_encoder = GGNN(3, hidden_size, hidden_size, direction_option=direction_option,
                                    use_edge_weight=use_edge_weight, dropout=feats_dropout)
        elif gnn == "GraphSage":
            self.gnn_encoder = GraphSAGE(3, hidden_size, hidden_size, hidden_size, aggregator_type="lstm",
                                         direction_option=direction_option, feat_drop=feats_dropout,
                                         activation=nn.ReLU(), bias=True, use_edge_weight=use_edge_weight)
        else:
            raise NotImplementedError()

        self.seq_decoder = StdRNNDecoder(max_decoder_step=50,
                                         decoder_input_size=2*hidden_size if direction_option == 'bi_sep' else hidden_size,
                                         decoder_hidden_size=hidden_size, graph_pooling_strategy=None,
                                         word_emb=self.word_emb, vocab=self.vocab.in_word_vocab,
                                         attention_type="sep_diff_encoder_type", fuse_strategy="concatenate",
                                         rnn_emb_input_size=hidden_size, use_coverage=True,
                                         tgt_emb_as_output_layer=True, dropout=rnn_dropout)
        self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)
        self.loss_cover = CoverageLoss(0.3)

    @classmethod
    def from_args(cls, vocab, args, device):
        return cls(vocab=vocab, hidden_size=args.hidden_size, graph_type=args.topology_type,
                   direction_option=args.gnn_direction, gnn=args.gnn, device=device,
                   emb_dropout=args.emb_dropout, feats_dropout=args.feats_dropout, rnn_dropout=args.feats_dropout)

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
