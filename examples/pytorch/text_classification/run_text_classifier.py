import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import dgl

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.datasets.trec import TrecDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN
from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda, EarlyStopping
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss


class TextClassifier(nn.Module):
    def __init__(self, vocab, config):
        super(TextClassifier, self).__init__()

        self.vocab = vocab
        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"}
        self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=config.num_hidden,
                                                               dropout=0.2,
                                                               fix_word_emb=False,
                                                               device=config.device)
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer


        heads = [config.num_heads] * (config.num_layers - 1) + [config.num_out_heads]
        self.gnn = GAT(config.num_layers,
                    config.num_hidden,
                    config.num_hidden,
                    config.num_hidden,
                    heads,
                    direction_option=config.direction_option,
                    feat_drop=config.in_drop,
                    attn_drop=config.attn_drop,
                    negative_slope=config.negative_slope,
                    residual=config.residual,
                    activation=F.elu)
        self.clf = FeedForwardNN(2 * config.num_hidden if config.direction_option == 'bi_sep' else config.num_hidden,
                        config.num_classes,
                        [config.num_hidden],
                        graph_pool_type=config.graph_pooling,
                        dim=config.num_hidden,
                        use_linear_proj=True)

        self.loss = GeneralLoss('CrossEntropy')


    def forward(self, graph_list, tgt=None, require_loss=True):
        # graph embedding construction
        batch_dgl_graph = self.graph_topology(graph_list)
        # convert DGLGraph to GraphData
        batch_graph = GraphData()
        batch_graph.from_dgl(batch_dgl_graph)

        # run GNN
        self.gnn(batch_graph)
        batch_dgl_graph.ndata['node_emb'] = batch_graph.node_features['node_emb']

        # run graph classifier
        self.clf(batch_dgl_graph)
        logits = batch_dgl_graph.graph_attributes['logits']

        if require_loss:
            loss = self.loss(logits, tgt)
            return logits, loss
        else:
            return logits


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = TrecDataset(root_dir="examples/pytorch/text_classification/data/trec",
                              topology_builder=DependencyBasedGraphConstruction,
                              topology_subdir='DependencyGraph',
                              pretrained_word_emb_file=self.config.pre_word_emb_file)
        data_size = len(dataset)
        self.train_dataloader = DataLoader(dataset[dataset.split_ids['train']], batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=1,
                                           collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset[dataset.split_ids['test']], batch_size=self.config.batch_size, shuffle=True,
                                          num_workers=1,
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model
        self.config.num_classes = dataset.num_classes

    def _build_model(self):
        self.model = TextClassifier(self.vocab, self.config).to(self.config.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config.lr)
        self.stopper = EarlyStopping('{}.{}'.format(self.config.save_model_path, self.config.seed), patience=self.config.patience)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config.lr_reduce_factor, \
            patience=self.config.lr_patience, verbose=True)

    def _build_evaluation(self):
        self.metric = Accuracy(['accuracy'])

    def train(self):
        max_score = -1
        dur = []
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = []
            train_acc = []
            t0 = time.time()
            for i, data in enumerate(self.train_dataloader):
                graph_list, tgt = data
                tgt = to_cuda(tgt, self.config.device)
                logits, loss = self.model(graph_list, tgt, require_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                pred = torch.max(logits, dim=-1)[1].cpu()
                train_acc.append(self.metric.calculate_scores(ground_truth=tgt.cpu(), predict=pred.cpu())[0])
                dur.append(time.time() - t0)

                if i > 10: # TODO
                    break

            val_acc = self.evaluate()
            self.scheduler.step(val_acc)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} | ValAcc {:.4f}".
              format(epoch, np.mean(dur), np.mean(train_loss), np.mean(train_acc), val_acc))

            if self.stopper.step(val_acc, self.model):
                break

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in self.test_dataloader:
                graph_list, tgt = data
                logits = self.model(graph_list, require_loss=False)
                pred_collect.append(logits)
                gt_collect.append(tgt)

            pred_collect = torch.max(torch.cat(pred_collect, 0), dim=-1)[1].cpu()
            gt_collect = torch.cat(gt_collect, 0).cpu()
            score = self.metric.calculate_scores(ground_truth=gt_collect, predict=pred_collect)[0]
            print("accuracy: {:.3f}".format(score))
            return score

def main(args):
    # Configure
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        print('[ Using CUDA ]')
        args.device = torch.device('cuda' if args.gpu < 0 else 'cuda:%d' % args.gpu)
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    else:
        args.device = torch.device('cpu')

    runner = ModelHandler(args)
    runner.train()
    print('Restored best saved model')
    runner.model = runner.stopper.load_checkpoint(runner.model)
    print('Removed best saved model file to save disk space')
    os.remove(runner.stopper.save_model_path)

    acc = runner.evaluate()
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='GAT')
    # register_data_args(parser)
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="use CPU")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use.")
    parser.add_argument("--graph_pooling", type=str, default='max_pool',
                        help="graph pooling (`avg_pool`, `max_pool`)")
    parser.add_argument("--direction_option", type=str, default='uni',
                        help="direction type (`uni`, `bi_fuse`, `bi_sep`)")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="learning rate patience")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5,
                        help="learning rate reduce factor")
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    # parser.add_argument('--dataset', type=str, default="",
    #                     help='dataset name')
    parser.add_argument('--pre_word_emb_file', type=str,
                        help='path to the pretrained word embedding file')
    parser.add_argument('--save_model_path', type=str, default="checkpoint",
                        help="path to the best saved model")

    args = parser.parse_args()

    main(args)

