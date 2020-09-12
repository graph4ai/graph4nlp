import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "5"

from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.datasets.cnn import CNNDataset
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import WordEmbedding
from graph4nlp.pytorch.modules.graph_embedding import *
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda
from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import warnings
warnings.filterwarnings("ignore")

class ExpressionAccuracy(EvaluationMetricBase):
    def __init__(self):
        super(ExpressionAccuracy, self).__init__()

    def calculate_scores(self, ground_truth, predict):
        correct = 0
        assert len(ground_truth) == len(predict)
        for gt, pred in zip(ground_truth, predict):
            print("ground truth: ", gt)
            print("prediction: ", pred)

            if gt == pred:
                correct += 1.
        return correct / len(ground_truth)


def logits2seq(prob: torch.Tensor):
    ids = prob.argmax(dim=-1)
    return ids


def wordid2str(word_ids, vocab: Vocab):
    ret = []
    assert len(word_ids.shape) == 2, print(word_ids.shape)
    for i in range(word_ids.shape[0]):
        id_list = word_ids[i, :]
        ret_inst = []
        for j in range(id_list.shape[0]):
            if id_list[j] == vocab.EOS:
                break
            token = vocab.getWord(id_list[j])
            ret_inst.append(token)
        ret.append(" ".join(ret_inst))
    return ret


def sequence_loss(logits, targets, xent_fn=None, pad_idx=0, if_aux=False, fp16=False):
    """ functional interface of SequenceLoss"""
    if if_aux:
        assert logits.size() == targets.size()
    else:
        assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask)
    if if_aux:
        target = target.float()
        logit = logits.masked_select(
            mask
        ).contiguous()
    else:
        logit = logits.masked_select(
            mask.unsqueeze(2).expand_as(logits)
        ).contiguous().view(-1, logits.size(-1))
    if xent_fn:
        if fp16:
            logit = torch.log(logit/(1-logit))
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


class Graph2seqLoss(nn.Module):
    def __init__(self, vocab: Vocab):
        super(Graph2seqLoss, self).__init__()
        self.VERY_SMALL_NUMBER = 1e-31
        self.vocab = vocab

    def forward(self, prob, gt):
        assert prob.shape[0:1] == gt.shape[0:1]
        assert len(prob.shape) == 3
        log_prob = torch.log(prob + self.VERY_SMALL_NUMBER)
        batch_size = gt.shape[0]
        step = gt.shape[1]

        mask = 1 - gt.data.eq(self.vocab.PAD).float()
        prob_select = torch.gather(log_prob.view(batch_size * step, -1), 1, gt.view(-1, 1))

        prob_select_masked = - torch.masked_select(prob_select, mask.view(-1, 1).byte())
        loss = torch.mean(prob_select_masked)
        return loss


class Graph2seq(nn.Module):
    def __init__(self, vocab, device, hidden_size=300, direction_option='bi_sep', parser='ie'):
        super(Graph2seq, self).__init__()
        self.device = device
        self.vocab = vocab
        self.parser = parser

        word_dropout = 0.2
        rnn_dropout = 0.2

        embedding_style = {'single_token_item': True if parser != 'ie' else False,
                           'emb_strategy': 'w2v_bilstm',
                           'num_rnn_layers': 1}

        graph_type = parser

        if graph_type == 'dependency':
            self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=hidden_size,
                                                                   word_dropout=word_dropout,
                                                                   rnn_dropout=rnn_dropout,
                                                                   fix_word_emb=False,
                                                                   fix_bert_emb=False,
                                                                   device=device)
        elif graph_type == 'constituency':
            self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=hidden_size,
                                                                   word_dropout=word_dropout,
                                                                   rnn_dropout=rnn_dropout,
                                                                   fix_word_emb=False,
                                                                   fix_bert_emb=False,
                                                                   device=device)
        elif graph_type == 'ie':
            self.graph_topology = IEBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=hidden_size,
                                                                   word_dropout=word_dropout,
                                                                   rnn_dropout=rnn_dropout,
                                                                   fix_word_emb=False,
                                                                   fix_bert_emb=False,
                                                                   device=device)
        # elif graph_type == 'node_emb':
        #     self.graph_topology = NodeEmbeddingBasedGraphConstruction(
        #                             vocab.in_word_vocab,
        #                             embedding_style,
        #                             sim_metric_type=config['gl_metric_type'],
        #                             num_heads=config['gl_num_heads'],
        #                             top_k_neigh=config['gl_top_k'],
        #                             epsilon_neigh=config['gl_epsilon'],
        #                             smoothness_ratio=config['gl_smoothness_ratio'],
        #                             connectivity_ratio=config['gl_connectivity_ratio'],
        #                             sparsity_ratio=config['gl_sparsity_ratio'],
        #                             input_size=config['num_hidden'],
        #                             hidden_size=config['gl_num_hidden'],
        #                             fix_word_emb=not config['no_fix_word_emb'],
        #                             fix_bert_emb=not config.get('no_fix_bert_emb', False),
        #                             word_dropout=config['word_dropout'],
        #                             rnn_dropout=config['rnn_dropout'],
        #                             device=config['device'])
        #     use_edge_weight = True
        # elif graph_type == 'node_emb_refined':
        #     self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
        #                             vocab.in_word_vocab,
        #                             embedding_style,
        #                             config['init_adj_alpha'],
        #                             sim_metric_type=config['gl_metric_type'],
        #                             num_heads=config['gl_num_heads'],
        #                             top_k_neigh=config['gl_top_k'],
        #                             epsilon_neigh=config['gl_epsilon'],
        #                             smoothness_ratio=config['gl_smoothness_ratio'],
        #                             connectivity_ratio=config['gl_connectivity_ratio'],
        #                             sparsity_ratio=config['gl_sparsity_ratio'],
        #                             input_size=config['num_hidden'],
        #                             hidden_size=config['gl_num_hidden'],
        #                             fix_word_emb=not config['no_fix_word_emb'],
        #                             fix_bert_emb=not config.get('no_fix_bert_emb', False),
        #                             word_dropout=config['word_dropout'],
        #                             rnn_dropout=config['rnn_dropout'],
        #                             device=config['device'])
        #     use_edge_weight = True
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(graph_type))

        if 'w2v' in self.graph_topology.embedding_layer.word_emb_layers:
            self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                            self.vocab.in_word_vocab.embeddings.shape[0],
                            self.vocab.in_word_vocab.embeddings.shape[1],
                            pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                            fix_emb=False,
                            device=device).word_emb_layer

        # self.gnn_encoder = GAT(2, hidden_size, hidden_size, hidden_size, [2, 1], direction_option=direction_option)
        self.gnn_encoder = GCN(2, hidden_size, hidden_size, hidden_size, direction_option=direction_option, activation=F.elu)

        self.seq_decoder = StdRNNDecoder(max_decoder_step=50,
                                         decoder_input_size=2 * hidden_size if direction_option == 'bi_sep' else hidden_size,
                                         decoder_hidden_size=hidden_size,
                                         word_emb=self.word_emb,
                                         vocab=self.vocab.in_word_vocab,
                                         attention_type="sep_diff_encoder_type",
                                         fuse_strategy="concatenate",
                                         rnn_emb_input_size=hidden_size,
                                         tgt_emb_as_output_layer=True,
                                         device=self.device)

        self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)

    def forward(self, graph_list, tgt=None, require_loss=True):
        batch_graph = self.graph_topology(graph_list)
        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features['rnn_emb'] = batch_graph.node_features['node_feat']

        # down-task
        prob, _, _ = self.seq_decoder(from_batch(batch_graph), tgt_seq=tgt)
        if require_loss:
            loss = self.loss_calc(prob, tgt)
            return prob, loss
        else:
            return prob


class CNN:
    def __init__(self, device=None, parser='ie'):
        super(CNN, self).__init__()
        self.device = device
        self.parser = parser
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        if self.parser == 'ie':
            graph_type = 'static'
            topology_builder = IEBasedGraphConstruction
            topology_subdir = 'IEGraph'
            dynamic_graph_type = None
            dynamic_init_topology_builder = None
        elif self.parser == 'dependency':
            graph_type = 'static'
            topology_builder = DependencyBasedGraphConstruction
            topology_subdir = 'DepGraph'
            dynamic_graph_type = None
            dynamic_init_topology_builder = None
        elif self.parser == 'node_emb':
            graph_type = 'dynamic'
            topology_builder = NodeEmbeddingBasedGraphConstruction
            topology_subdir = 'NodeEmb'
            dynamic_graph_type = 'node_emb'
            dynamic_init_topology_builder = DependencyBasedGraphConstruction
        else:
            raise NotImplementedError()

        dataset = CNNDataset(root_dir="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn",
                             graph_type=graph_type,
                             topology_builder=topology_builder,
                             topology_subdir=topology_subdir,
                             dynamic_graph_type=dynamic_graph_type,
                             dynamic_init_topology_builder=dynamic_init_topology_builder,
                             dynamic_init_topology_aux_args={'dummy_param': 0})

        self.train_dataloader = DataLoader(dataset.train, batch_size=32,
                                           shuffle=True, num_workers=1,
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=32,
                                         shuffle=True, num_workers=1,
                                         collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=32,
                                          shuffle=True, num_workers=1,
                                          collate_fn=dataset.collate_fn)

        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = to_cuda(Graph2seq(self.vocab, device=self.device, parser=self.parser), self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=1e-3)

    def _build_evaluation(self):
        self.metrics = [ROUGE()]

    def train(self, max_score_prev=-1):
        max_score = max_score_prev
        for epoch in range(200):
            self.model.train()
            print("Epoch: {}".format(epoch))
            for data in self.train_dataloader:
                graph_list, tgt = data
                tgt = to_cuda(tgt, self.device)
                _, loss = self.model(graph_list, tgt, require_loss=True)
                print(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch > 1:
                score = self.evaluate(self.val_dataloader)
                if score > max_score:
                    torch.save(self.model.state_dict(), 'best_cnn_model.pt')
                max_score = max(max_score, score)
        return max_score

    def evaluate(self, dataloader, test_mode=False):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        for data in dataloader:
            graph_list, tgt = data
            prob = self.model(graph_list, require_loss=False)
            pred = logits2seq(prob)

            pred_str = wordid2str(pred.detach().cpu(), self.vocab.in_word_vocab)
            tgt_str = wordid2str(tgt, self.vocab.in_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        if test_mode==True:
            with open('cnn_pred_output.txt','w+') as f:
                for line in pred_collect:
                    f.write(line+'\n')

            with open('cnn_tgt_output.txt','w+') as f:
                for line in gt_collect:
                    f.write(line+'\n')

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)[0]
        print("rouge: {:.3f}".format(score))
        return score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--parser', type=str, default='ie')
    parser.add_argument('--load_model', type=str, default='True')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('using cuda...')
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    runner = CNN(device, args.parser)

    if args.load_model == 'True':
        runner.model.load_state_dict(
            torch.load('/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/best_cnn_model.pt',
                       map_location=device))
        max_score_prev = runner.evaluate(runner.val_dataloader)
    else:
        max_score_prev = -1
    # max_score = runner.train(max_score_prev)
    # print("Train finish, best score: {:.3f}".format(max_score))
    runner.model.load_state_dict(torch.load('best_cnn_model.pt'))
    test_score = runner.evaluate(runner.test_dataloader, test_mode=True)
    print("Test score: {:.3f}".format(test_score))

# 0.14769418484038643(1)  0.050071791802983635(2)  0.1431397271091325(l)