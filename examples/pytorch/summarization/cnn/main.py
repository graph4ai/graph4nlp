import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from .dataset import CNNDataset
from .model import Graph2seq
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from .config import get_args
from .utils import get_log, wordid2str
from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
import time


class CNN:
    def __init__(self, opt):
        super(CNN, self).__init__()
        self.opt = opt
        self._build_device(self.opt)
        self._build_logger(self.opt.log_file)
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_device(self, opt):
        seed = opt.seed
        np.random.seed(seed)
        if opt.use_gpu != 0 and torch.cuda.is_available():
            print('[ Using CUDA ]')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn
            cudnn.benchmark = True
            device = torch.device('cuda' if opt.gpu < 0 else 'cuda:%d' % opt.gpu)
        else:
            print('[ Using CPU ]')
            device = torch.device('cpu')
        self.device = device

    def _build_logger(self, log_file):
        self.logger = get_log(log_file)

    def _build_dataloader(self):

        # dataset = CNNDataset(root_dir=self.opt.root_dir,
        #                              topology_builder=DependencyBasedGraphConstruction,
        #                              topology_subdir=self.opt.topology_subdir, share_vocab=False,
        #                              word_emb_size=self.opt.word_emb_size)

        if self.opt.topology_subdir == 'ie':
            graph_type = 'static'
            topology_builder = IEBasedGraphConstruction
            topology_subdir = 'IEGraph'
            dynamic_graph_type = None
            dynamic_init_topology_builder = None
        elif self.opt.topology_subdir == 'DependencyGraph':
            graph_type = 'static'
            topology_builder = DependencyBasedGraphConstruction
            topology_subdir = 'DependencyGraph'
            dynamic_graph_type = None
            dynamic_init_topology_builder = None
        elif self.opt.topology_subdir == 'node_emb':
            graph_type = 'dynamic'
            topology_builder = NodeEmbeddingBasedGraphConstruction
            topology_subdir = 'NodeEmb'
            dynamic_graph_type = 'node_emb'
            dynamic_init_topology_builder = DependencyBasedGraphConstruction
        else:
            raise NotImplementedError()

        dataset = CNNDataset(root_dir=self.opt.root_dir,
                             graph_type=graph_type,
                             topology_builder=topology_builder,
                             topology_subdir=topology_subdir,
                             dynamic_graph_type=dynamic_graph_type,
                             dynamic_init_topology_builder=dynamic_init_topology_builder,
                             dynamic_init_topology_aux_args={'dummy_param': 0})

        self.train_dataloader = DataLoader(dataset.train, batch_size=self.opt.batch_size, shuffle=True, num_workers=10,
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=self.opt.batch_size, shuffle=False, num_workers=10,
                                         collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.opt.batch_size, shuffle=False, num_workers=10,
                                          collate_fn=dataset.collate_fn)
        self.vocab: VocabModel = dataset.vocab_model

        # import torchtext.vocab as vocab
        # glove = vocab.GloVe
        # glove.url["de"] = "/home/shiina/shiina/lib/graph4nlp/.vector_cache/glove.de.300d.txt"
        # from .utils import get_glove_weights
        # en = glove(name='6B', dim=300)
        #
        # pretrained_weight = get_glove_weights(en, self.vocab.in_word_vocab)
        # self.vocab.in_word_vocab.embeddings = pretrained_weight.numpy()
        # print("English word embedding loaded")
        #
        # de = glove(name='de', dim=300)
        #
        # pretrained_weight = get_glove_weights(de, self.vocab.out_word_vocab)
        # self.vocab.out_word_vocab.embeddings = pretrained_weight.numpy()
        # print("De word embedding loaded")

    def _build_model(self):
        self.model = Graph2seq(self.vocab, gnn=self.opt.gnn, device=self.device,
                               rnn_dropout=self.opt.rnn_dropout, word_dropout=self.opt.word_dropout,
                               hidden_size=self.opt.hidden_size,
                               word_emb_size=self.opt.word_emb_size).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt.learning_rate)

    def _build_evaluation(self):
        self.metrics = [ROUGE()]

    def train(self):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(200):
            self.model.train()
            self.train_epoch(epoch, split="train")
            self._adjust_lr(epoch)
            if epoch >= 0:
                score = self.evaluate(split="val")
                if score >= max_score:
                    self.logger.info("Best model saved, epoch {}".format(epoch))
                    self.save_checkpoint("best.pth")
                    self._best_epoch = epoch
                max_score = max(max_score, score)
                score = self.evaluate(split="test")
            if epoch >= 30 and self._stop_condition(epoch):
                break
        return max_score

    def _stop_condition(self, epoch, patience=2000):
        return epoch > patience + self._best_epoch

    def _adjust_lr(self, epoch):
        def set_lr(optimizer, decay_factor):
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * decay_factor

        epoch_diff = epoch - self.opt.lr_start_decay_epoch
        if epoch_diff >= 0 and epoch_diff % self.opt.lr_decay_per_epoch == 0:
            if self.opt.learning_rate > self.opt.min_lr:
                set_lr(self.optimizer, self.opt.lr_decay_rate)
                self.opt.learning_rate = self.opt.learning_rate * self.opt.lr_decay_rate
                self.logger.info("Learning rate adjusted: {:.5f}".format(self.opt.learning_rate))

    def train_epoch(self, epoch, split="train"):
        assert split in ["train"]
        self.logger.info("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_dataloader
        step_all_train = len(dataloader)
        start = time.time()
        for step, data in enumerate(dataloader):
            graph_list, tgt = data
            tgt = tgt.to(self.device)
            _, loss = self.model(graph_list, tgt, require_loss=True)
            loss_collect.append(loss.item())
            if step % self.opt.loss_display_step == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "Epoch {}: [{} / {}] loss: {:.3f}, time cost: {:.3f}".format(epoch, step, step_all_train,
                                                                                 np.mean(loss_collect), end - start))
                start = time.time()
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            graph_list, tgt = data
            prob = self.model(graph_list, require_loss=False)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), self.vocab.out_word_vocab)
            tgt_str = wordid2str(tgt, self.vocab.out_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        score, _ = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation results in `{}` split".format(split))
        self.logger.info("ROUGE: {:.3f}".format(score))
        return score

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    runner = CNN(opt)
    max_score = runner.train()
    print("Train finish, best val score: {:.3f}".format(max_score))
    runner.load_checkpoint("best_all.pth")
    runner.evaluate(split="test")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# # os.environ['CUDA_LAUNCH_BLOCKING'] = "5"
#
# from graph4nlp.pytorch.data.data import from_batch
# # from graph4nlp.pytorch.datasets.cnn import CNNDataset
# from examples.pytorch.summarization.cnn.dataset import CNNDataset
# from graph4nlp.pytorch.modules.graph_construction import *
# from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
# from graph4nlp.pytorch.modules.graph_construction.embedding_construction import WordEmbedding
# from graph4nlp.pytorch.modules.graph_embedding import *
# from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
# from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
# from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda
# from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
#
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import math
#
# import warnings
# warnings.filterwarnings("ignore")
#
# class ExpressionAccuracy(EvaluationMetricBase):
#     def __init__(self):
#         super(ExpressionAccuracy, self).__init__()
#
#     def calculate_scores(self, ground_truth, predict):
#         correct = 0
#         assert len(ground_truth) == len(predict)
#         for gt, pred in zip(ground_truth, predict):
#             print("ground truth: ", gt)
#             print("prediction: ", pred)
#
#             if gt == pred:
#                 correct += 1.
#         return correct / len(ground_truth)
#
#
# def logits2seq(prob: torch.Tensor):
#     ids = prob.argmax(dim=-1)
#     return ids
#
#
# def wordid2str(word_ids, vocab: Vocab):
#     ret = []
#     assert len(word_ids.shape) == 2, print(word_ids.shape)
#     for i in range(word_ids.shape[0]):
#         id_list = word_ids[i, :]
#         ret_inst = []
#         for j in range(id_list.shape[0]):
#             if id_list[j] == vocab.EOS:
#                 break
#             token = vocab.getWord(id_list[j])
#             ret_inst.append(token)
#         ret.append(" ".join(ret_inst))
#     return ret
#
#
# def sequence_loss(logits, targets, xent_fn=None, pad_idx=0, if_aux=False, fp16=False):
#     """ functional interface of SequenceLoss"""
#     if if_aux:
#         assert logits.size() == targets.size()
#     else:
#         assert logits.size()[:-1] == targets.size()
#
#     mask = targets != pad_idx
#     target = targets.masked_select(mask)
#     if if_aux:
#         target = target.float()
#         logit = logits.masked_select(
#             mask
#         ).contiguous()
#     else:
#         logit = logits.masked_select(
#             mask.unsqueeze(2).expand_as(logits)
#         ).contiguous().view(-1, logits.size(-1))
#     if xent_fn:
#         if fp16:
#             logit = torch.log(logit/(1-logit))
#         loss = xent_fn(logit, target)
#     else:
#         loss = F.cross_entropy(logit, target)
#     assert (not math.isnan(loss.mean().item())
#             and not math.isinf(loss.mean().item()))
#     return loss
#
#
# class Graph2seqLoss(nn.Module):
#     def __init__(self, vocab: Vocab):
#         super(Graph2seqLoss, self).__init__()
#         self.VERY_SMALL_NUMBER = 1e-31
#         self.vocab = vocab
#
#     def forward(self, prob, gt):
#         assert prob.shape[0:1] == gt.shape[0:1]
#         assert len(prob.shape) == 3
#         log_prob = torch.log(prob + self.VERY_SMALL_NUMBER)
#         batch_size = gt.shape[0]
#         step = gt.shape[1]
#
#         mask = 1 - gt.data.eq(self.vocab.PAD).float()
#         prob_select = torch.gather(log_prob.view(batch_size * step, -1), 1, gt.view(-1, 1))
#
#         prob_select_masked = - torch.masked_select(prob_select, mask.view(-1, 1).byte())
#         loss = torch.mean(prob_select_masked)
#         return loss
#
#
# class Graph2seq(nn.Module):
#     def __init__(self, vocab, device, hidden_size=300, direction_option='bi_sep', parser='ie'):
#         super(Graph2seq, self).__init__()
#         self.device = device
#         self.vocab = vocab
#         self.parser = parser
#
#         word_dropout = 0.2
#         rnn_dropout = 0.2
#
#         embedding_style = {'single_token_item': True if parser != 'ie' else False,
#                            'emb_strategy': 'w2v_bilstm',
#                            'num_rnn_layers': 1}
#
#         graph_type = parser
#
#         if graph_type == 'dependency':
#             self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
#                                                                    vocab=vocab.in_word_vocab,
#                                                                    hidden_size=hidden_size,
#                                                                    word_dropout=word_dropout,
#                                                                    rnn_dropout=rnn_dropout,
#                                                                    fix_word_emb=False,
#                                                                    fix_bert_emb=False,
#                                                                    device=device)
#         elif graph_type == 'constituency':
#             self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
#                                                                    vocab=vocab.in_word_vocab,
#                                                                    hidden_size=hidden_size,
#                                                                    word_dropout=word_dropout,
#                                                                    rnn_dropout=rnn_dropout,
#                                                                    fix_word_emb=False,
#                                                                    fix_bert_emb=False,
#                                                                    device=device)
#         elif graph_type == 'ie':
#             self.graph_topology = IEBasedGraphConstruction(embedding_style=embedding_style,
#                                                                    vocab=vocab.in_word_vocab,
#                                                                    hidden_size=hidden_size,
#                                                                    word_dropout=word_dropout,
#                                                                    rnn_dropout=rnn_dropout,
#                                                                    fix_word_emb=False,
#                                                                    fix_bert_emb=False,
#                                                                    device=device)
#         # elif graph_type == 'node_emb':
#         #     self.graph_topology = NodeEmbeddingBasedGraphConstruction(
#         #                             vocab.in_word_vocab,
#         #                             embedding_style,
#         #                             sim_metric_type=config['gl_metric_type'],
#         #                             num_heads=config['gl_num_heads'],
#         #                             top_k_neigh=config['gl_top_k'],
#         #                             epsilon_neigh=config['gl_epsilon'],
#         #                             smoothness_ratio=config['gl_smoothness_ratio'],
#         #                             connectivity_ratio=config['gl_connectivity_ratio'],
#         #                             sparsity_ratio=config['gl_sparsity_ratio'],
#         #                             input_size=config['num_hidden'],
#         #                             hidden_size=config['gl_num_hidden'],
#         #                             fix_word_emb=not config['no_fix_word_emb'],
#         #                             fix_bert_emb=not config.get('no_fix_bert_emb', False),
#         #                             word_dropout=config['word_dropout'],
#         #                             rnn_dropout=config['rnn_dropout'],
#         #                             device=config['device'])
#         #     use_edge_weight = True
#         # elif graph_type == 'node_emb_refined':
#         #     self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
#         #                             vocab.in_word_vocab,
#         #                             embedding_style,
#         #                             config['init_adj_alpha'],
#         #                             sim_metric_type=config['gl_metric_type'],
#         #                             num_heads=config['gl_num_heads'],
#         #                             top_k_neigh=config['gl_top_k'],
#         #                             epsilon_neigh=config['gl_epsilon'],
#         #                             smoothness_ratio=config['gl_smoothness_ratio'],
#         #                             connectivity_ratio=config['gl_connectivity_ratio'],
#         #                             sparsity_ratio=config['gl_sparsity_ratio'],
#         #                             input_size=config['num_hidden'],
#         #                             hidden_size=config['gl_num_hidden'],
#         #                             fix_word_emb=not config['no_fix_word_emb'],
#         #                             fix_bert_emb=not config.get('no_fix_bert_emb', False),
#         #                             word_dropout=config['word_dropout'],
#         #                             rnn_dropout=config['rnn_dropout'],
#         #                             device=config['device'])
#         #     use_edge_weight = True
#         else:
#             raise RuntimeError('Unknown graph_type: {}'.format(graph_type))
#
#         if 'w2v' in self.graph_topology.embedding_layer.word_emb_layers:
#             self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
#         else:
#             self.word_emb = WordEmbedding(
#                             self.vocab.in_word_vocab.embeddings.shape[0],
#                             self.vocab.in_word_vocab.embeddings.shape[1],
#                             pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
#                             fix_emb=False,
#                             device=device).word_emb_layer
#
#         self.gnn_encoder = GAT(2, hidden_size, hidden_size, hidden_size, [2, 1], direction_option=direction_option)
#         # self.gnn_encoder = GCN(2, hidden_size, hidden_size, hidden_size, direction_option=direction_option, activation=F.elu)
#
#         self.seq_decoder = StdRNNDecoder(max_decoder_step=50,
#                                          decoder_input_size=2 * hidden_size if direction_option == 'bi_sep' else hidden_size,
#                                          decoder_hidden_size=hidden_size,
#                                          word_emb=self.word_emb,
#                                          vocab=self.vocab.in_word_vocab,
#                                          attention_type="sep_diff_encoder_type",
#                                          fuse_strategy="concatenate",
#                                          rnn_emb_input_size=hidden_size,
#                                          tgt_emb_as_output_layer=True,
#                                          device=self.device)
#
#         self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)
#
#     def forward(self, graph_list, tgt=None, require_loss=True):
#         batch_graph = self.graph_topology(graph_list)
#         batch_graph = self.gnn_encoder(batch_graph)
#         batch_graph.node_features['rnn_emb'] = batch_graph.node_features['node_feat']
#
#         # down-task
#         prob, _, _ = self.seq_decoder(from_batch(batch_graph), tgt_seq=tgt)
#         if require_loss:
#             loss = self.loss_calc(prob, tgt)
#             return prob, loss
#         else:
#             return prob
#
#
# class CNN:
#     def __init__(self, device=None, parser='ie'):
#         super(CNN, self).__init__()
#         self.device = device
#         self.parser = parser
#         self._build_dataloader()
#         self._build_model()
#         self._build_optimizer()
#         self._build_evaluation()
#
#     def _build_dataloader(self):
#         if self.parser == 'ie':
#             graph_type = 'static'
#             topology_builder = IEBasedGraphConstruction
#             topology_subdir = 'IEGraph'
#             dynamic_graph_type = None
#             dynamic_init_topology_builder = None
#         elif self.parser == 'dependency':
#             graph_type = 'static'
#             topology_builder = DependencyBasedGraphConstruction
#             topology_subdir = 'DependencyGraph'
#             # topology_subdir = 'DepGraph'
#             dynamic_graph_type = None
#             dynamic_init_topology_builder = None
#         elif self.parser == 'node_emb':
#             graph_type = 'dynamic'
#             topology_builder = NodeEmbeddingBasedGraphConstruction
#             topology_subdir = 'NodeEmb'
#             dynamic_graph_type = 'node_emb'
#             dynamic_init_topology_builder = DependencyBasedGraphConstruction
#         else:
#             raise NotImplementedError()
#
#         dataset = CNNDataset(root_dir="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn",
#                              graph_type=graph_type,
#                              topology_builder=topology_builder,
#                              topology_subdir=topology_subdir,
#                              dynamic_graph_type=dynamic_graph_type,
#                              dynamic_init_topology_builder=dynamic_init_topology_builder,
#                              dynamic_init_topology_aux_args={'dummy_param': 0})
#
#         self.train_dataloader = DataLoader(dataset.train, batch_size=128,
#                                            shuffle=True, num_workers=0,
#                                            collate_fn=dataset.collate_fn)
#         self.val_dataloader = DataLoader(dataset.val, batch_size=64,
#                                          shuffle=True, num_workers=0,
#                                          collate_fn=dataset.collate_fn)
#         self.test_dataloader = DataLoader(dataset.test, batch_size=64,
#                                           shuffle=True, num_workers=0,
#                                           collate_fn=dataset.collate_fn)
#
#         self.vocab = dataset.vocab_model
#
#     def _build_model(self):
#         self.model = to_cuda(Graph2seq(self.vocab, device=self.device, parser=self.parser), self.device)
#
#     def _build_optimizer(self):
#         parameters = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = optim.Adam(parameters, lr=1e-3)
#
#     def _build_evaluation(self):
#         self.metrics = [ROUGE()]
#
#     def train(self, max_score_prev=-1):
#         max_score = max_score_prev
#         print_info = []
#         for epoch in range(200):
#             self.model.train()
#             print_info.append("Epoch: {}".format(epoch))
#             for data in self.train_dataloader:
#                 graph_list, tgt = data
#                 tgt = to_cuda(tgt, self.device)
#                 _, loss = self.model(graph_list, tgt, require_loss=True)
#                 print_info.append(loss.item())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#             if epoch >= 0:
#                 score = self.evaluate(self.val_dataloader)
#                 print_info.append(score)
#                 if score > max_score:
#                     torch.save(self.model.state_dict(), '/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/best_cnn_model_all.pt')
#                 max_score = max(max_score, score)
#
#                 with open('/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/print_info.log', 'w') as f:
#                     for line in print_info:
#                         f.write(str(line)+'\n')
#         return max_score
#
#     def evaluate(self, dataloader, test_mode=False):
#         self.model.eval()
#         pred_collect = []
#         gt_collect = []
#         for data in dataloader:
#             graph_list, tgt = data
#             prob = self.model(graph_list, require_loss=False)
#             pred = logits2seq(prob)
#
#             pred_str = wordid2str(pred.detach().cpu(), self.vocab.in_word_vocab)
#             tgt_str = wordid2str(tgt, self.vocab.in_word_vocab)
#             pred_collect.extend(pred_str)
#             gt_collect.extend(tgt_str)
#
#         if test_mode==True:
#             with open('cnn_pred_output.txt','w+') as f:
#                 for line in pred_collect:
#                     f.write(line+'\n')
#
#             with open('cnn_tgt_output.txt','w+') as f:
#                 for line in gt_collect:
#                     f.write(line+'\n')
#
#         score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)[0]
#         print("rouge: {:.3f}".format(score))
#         return score
#
#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--parser', type=str, default='ie')
#     parser.add_argument('--load_model', type=str, default='True')
#     args = parser.parse_args()
#
#     if torch.cuda.is_available():
#         print('using cuda...')
#         device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu')
#     runner = CNN(device, args.parser)
#
#     if args.load_model == 'True':
#         runner.model.load_state_dict(
#             torch.load('/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/best_cnn_model_all.pt',
#                        map_location=device))
#         max_score_prev = runner.evaluate(runner.val_dataloader)
#     else:
#         max_score_prev = -1
#     max_score = runner.train(max_score_prev)
#     print("Train finish, best score: {:.3f}".format(max_score))
#     runner.model.load_state_dict(torch.load('best_cnn_model.pt'))
#     test_score = runner.evaluate(runner.test_dataloader, test_mode=True)
#     print("Test score: {:.3f}".format(test_score))
#
# # 0.14769418484038643(1)  0.050071791802983635(2)  0.1431397271091325(l)