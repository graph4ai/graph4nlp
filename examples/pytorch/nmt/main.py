import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import dgl

from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
from graph4nlp.pytorch.test.example.syntax_nmt.dataset import EuroparlNMTDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction

class Graph2seq(nn.Module):
    def __init__(self, vocab, hidden_size=300, direction_option='bi_sep'):
        super(Graph2seq, self).__init__()

        self.vocab = vocab
        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"}
        self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=hidden_size, dropout=0.2, use_cuda=True,
                                                               fix_word_emb=False)
        self.gnn = None
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer
        self.gnn_encoder = GAT(2, hidden_size, hidden_size, hidden_size, [2, 1], direction_option=direction_option)
        self.seq_decoder = StdRNNDecoder(max_decoder_step=50,
                                         decoder_input_size=2 * hidden_size if direction_option == 'bi_sep' else hidden_size,
                                         decoder_hidden_size=hidden_size,
                                         word_emb=self.word_emb, vocab=self.vocab.in_word_vocab,
                                         attention_type="sep_diff_encoder_type", fuse_strategy="concatenate",
                                         rnn_emb_input_size=hidden_size,
                                         tgt_emb_as_output_layer=True, device=self.graph_topology.device)
        self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)

    def forward(self, graph_list, tgt=None, require_loss=True):
        batch_dgl_graph = self.graph_topology(graph_list)
        # do graph nn here
        # convert DGLGraph to GraphData
        batch_graph = GraphData()
        batch_graph.from_dgl(batch_dgl_graph)

        # run GNN
        batch_graph = self.gnn_encoder(batch_graph)
        batch_dgl_graph.ndata['node_emb'] = batch_graph.node_features['node_emb']
        batch_dgl_graph.ndata['rnn_emb'] = batch_graph.node_features['node_feat']

        dgl_graph_list = dgl.unbatch(batch_dgl_graph)
        for g, dg in zip(graph_list, dgl_graph_list):
            g.node_features["node_emb"] = dg.ndata["node_emb"]
            g.node_features["rnn_emb"] = dg.ndata["rnn_emb"]

        # down-task
        prob, _, _ = self.seq_decoder(graph_list, tgt_seq=tgt)
        if require_loss:
            loss = self.loss_calc(prob, tgt)
            return prob, loss
        else:
            return prob


class NMT:
    def __init__(self):
        super(NMT, self).__init__()
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = EuroparlNMTDataset(root_dir="/home/shiina/shiina/lib/dataset",
                                 topology_builder=DependencyBasedGraphConstruction,
                                 topology_subdir='DependencyGraph')
        data_size = len(dataset)
        self.train_dataloader = DataLoader(dataset[:int(0.8 * data_size)], batch_size=24, shuffle=True,
                                           num_workers=10,
                                           collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset[int(0.8 * data_size):], batch_size=24, shuffle=False,
                                          num_workers=10,
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Graph2seq(self.vocab).cuda()

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=1e-3)

    def _build_evaluation(self):
        self.metrics = []

    def train(self):
        max_score = -1
        for epoch in range(200):
            self.model.train()
            print("Epoch: {}".format(epoch))
            for data in self.train_dataloader:
                graph_list, tgt = data
                tgt = tgt.cuda()
                _, loss = self.model(graph_list, tgt, require_loss=True)
                print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch > 4:
                score = self.evaluate()
                max_score = max(max_score, score)
        return max_score

    def evaluate(self):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        for data in self.test_dataloader:
            graph_list, tgt = data
            prob = self.model(graph_list, require_loss=False)
            pred = logits2seq(prob)

            pred_str = wordid2str(pred.detach().cpu(), self.vocab.in_word_vocab)
            tgt_str = wordid2str(tgt, self.vocab.in_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("accuracy: {:.3f}".format(score))
        return score

