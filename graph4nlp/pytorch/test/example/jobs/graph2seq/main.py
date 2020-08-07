import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import dgl

from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy


def logits2seq(prob: torch.Tensor):
    ids = prob.argmax(dim=-1)
    return ids


class Graph2seqLoss(nn.Module):
    def __init__(self):
        super(Graph2seqLoss, self).__init__()
        self.loss_func = nn.NLLLoss()
        self.VERY_SMALL_NUMBER = 1e-31

    def forward(self, prob, gt):
        assert prob.shape[0:1] == gt.shape[0:1]
        assert len(prob.shape) == 3
        log_prob = torch.log(prob + self.VERY_SMALL_NUMBER)
        loss = self.loss_func(log_prob.view(-1, prob.shape[2]), gt.view(-1))
        return loss

class Graph2seq(nn.Module):
    def __init__(self, vocab, hidden_size=300):
        super(Graph2seq, self).__init__()

        self.vocab = vocab
        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "bilstm", 'seq_info_encode_strategy': "bilstm"}
        self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style, vocab=vocab.word_vocab,
                                                               hidden_size=hidden_size, dropout=0.2, use_cuda=True, fix_word_emb=False)
        self.gnn = None
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer
        self.gnn_encoder = GAT(2, hidden_size, hidden_size, hidden_size, [2, 1], direction_option='bi_fuse')
        self.seq_decoder = StdRNNDecoder(max_decoder_step=50, decoder_input_size=hidden_size, decoder_hidden_size=hidden_size,
                                         word_emb=self.word_emb, vocab=self.vocab.word_vocab,
                                         tgt_emb_as_output_layer=True, device=self.graph_topology.device)
        self.loss_calc = Graph2seqLoss()

    def forward(self, graph_list, tgt=None, require_loss=True):
        batch_dgl_graph = self.graph_topology(graph_list)
        # do graph nn here
        # convert DGLGraph to GraphData
        batch_graph = GraphData()
        batch_graph.from_dgl(batch_dgl_graph)

        # run GNN
        batch_graph = self.gnn_encoder(batch_graph)
        batch_dgl_graph.ndata['node_emb'] = batch_graph.node_features['node_emb']

        dgl_graph_list = dgl.unbatch(batch_dgl_graph)
        for g, dg in zip(graph_list, dgl_graph_list):
            g.node_features["node_emb"] = dg.ndata["node_emb"]

        # down-task
        prob, _, _ = self.seq_decoder(graph_list, tgt_seq=tgt)
        if require_loss:
            loss = self.loss_calc(prob, tgt)
            return prob, loss
        else:
            return prob

class Jobs:
    def __init__(self):
        super(Jobs, self).__init__()
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()


    def _build_dataloader(self):
        train_dataset = JobsDataset(root_dir="graph4nlp/pytorch/test/dataset/jobs")
        # train_dataset = JobsDataset(root_dir="../../../dataset/jobs")
        self.train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=1,
                                           collate_fn=train_dataset.collate_fn)
        self.vocab = train_dataset.vocab_model

    def _build_model(self):
        self.model = Graph2seq(self.vocab).cuda()

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=1e-4)

    def _build_evaluation(self):
        self.metrics = [Accuracy(metrics=["accuracy"])]

    def train(self):
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
            if epoch > 10:
                self.evaluate()

        pass

    def evaluate(self):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        for data in self.train_dataloader:
            graph_list, tgt = data
            prob = self.model(graph_list, require_loss=False)
            pred = logits2seq(prob)
            pred_collect.append(pred.detach().cpu())
            gt_collect.append(tgt)
        ground_truth = torch.cat(gt_collect, dim=0).view(-1)
        predict = torch.cat(pred_collect, dim=0).view(-1)
        for evaluation_method in self.metrics:
            score = evaluation_method.calculate_scores(ground_truth=ground_truth, predict=predict)
            print("accuracy: ", score)


if __name__ == "__main__":
    runner = Jobs()
    runner.train()
    runner.evaluate()
