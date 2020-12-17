import os
import time
import datetime
import copy
import argparse
from argparse import Namespace
import yaml

import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

from graph4nlp.pytorch.datasets.squad import SQuADDataset
from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import RNNEmbedding, WordEmbedding
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.utils.generic_utils import grid, to_cuda, dropout_fn, sparse_mx_to_torch_sparse_tensor, EarlyStopping
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.evaluation import BLEU, METEOR, ROUGE
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import DecoderStrategy
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config
import multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
multiprocessing.set_start_method("spawn", force=True)


class QGModel(nn.Module):
    def __init__(self, vocab, config):
        super(QGModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.use_coverage = self.config['decoder_args']['rnn_decoder_share']['use_coverage']

        # build Graph2Seq model
        self.g2s = Graph2Seq.from_args(config, self.vocab, config['device'])

        if 'w2v' in self.g2s.graph_topology.embedding_layer.word_emb_layers:
            self.word_emb = self.g2s.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                            self.vocab.in_word_vocab.embeddings.shape[0],
                            self.vocab.in_word_vocab.embeddings.shape[1],
                            pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                            fix_emb=config['graph_construction_args']['node_embedding']['fix_word_emb'],
                            device=config['device']).word_emb_layer

        answer_feat_size = self.vocab.in_word_vocab.embeddings.shape[1]
        if 'node_edge_bert' in self.g2s.graph_topology.embedding_layer.word_emb_layers:
            self.bert_encoder = self.g2s.graph_topology.embedding_layer.word_emb_layers['node_edge_bert']
            answer_feat_size += self.bert_encoder.bert_model.config.hidden_size
        elif 'seq_bert' in self.g2s.graph_topology.embedding_layer.word_emb_layers:
            self.bert_encoder = self.g2s.graph_topology.embedding_layer.word_emb_layers['seq_bert']
            answer_feat_size += self.bert_encoder.bert_model.config.hidden_size
        else:
            self.bert_encoder = None


        self.ans_rnn_encoder = RNNEmbedding(
                                    answer_feat_size,
                                    config['num_hidden'],
                                    bidirectional=True,
                                    num_layers=1,
                                    rnn_type='lstm',
                                    dropout=config['enc_rnn_dropout'],
                                    device=config['device'])

        # soft-alignment between context and answer
        self.ctx2ans_attn = Context2AnswerAttention(config['num_hidden'], config['num_hidden'])
        self.fuse_ctx_ans = nn.Linear(2 * config['num_hidden'], config['num_hidden'], bias=False)

        self.loss_calc = Graph2SeqLoss(ignore_index=self.vocab.out_word_vocab.PAD,
                                  use_coverage=self.use_coverage, coverage_weight=config['coverage_loss_ratio'])

    def encode_init_node_feature(self, data):
        num_graph_nodes = []
        for g in data['graph_data']:
            num_graph_nodes.append(g.get_node_num())

        # graph embedding construction
        batch_gd = self.g2s.graph_topology(data['graph_data'])

        # answer alignment
        answer_feat = self.word_emb(data['input_tensor2'])
        answer_feat = dropout_fn(answer_feat, self.config['word_dropout'], shared_axes=[-2], training=self.training)

        if self.bert_encoder is not None:
            answer_bert_feat = self.bert_encoder(data['input_text2'])
            answer_feat = torch.cat([answer_feat, answer_bert_feat], -1)

        answer_feat = self.ans_rnn_encoder(answer_feat, data['input_length2'])[0]
        new_node_feat = self.answer_alignment(batch_gd.node_features['node_feat'], answer_feat, answer_feat, num_graph_nodes, data['input_length2'])
        batch_gd.node_features['node_feat'] = new_node_feat

        return batch_gd

    def forward(self, data, oov_dict=None, require_loss=True):
        batch_gd = self.encode_init_node_feature(data)
        prob, enc_attn_weights, coverage_vectors = self.g2s.encoder_decoder(batch_gd, data['graph_data'], oov_dict=oov_dict, tgt_seq=data['tgt_tensor'])

        if require_loss:
            loss = self.loss_calc(prob, label=data['tgt_tensor'], enc_attn_weights=enc_attn_weights, coverage_vectors=coverage_vectors)
            return prob, loss
        else:
            return prob

    def answer_alignment(self, node_feat, answer_feat, out_answer_feat, num_nodes, answer_length):
        assert len(num_nodes) == len(answer_length)
        mask = []
        for i in range(len(num_nodes)): # batch
            tmp_mask = np.ones((num_nodes[i], answer_feat.shape[1]))
            tmp_mask[:, answer_length[i]:] = 0
            mask.append(sparse.coo_matrix(tmp_mask))

        mask = sparse.block_diag(mask)
        mask = to_cuda(sparse_mx_to_torch_sparse_tensor(mask).to_dense(), self.config['device'])

        answer_feat = answer_feat.reshape(-1, answer_feat.shape[-1])
        out_answer_feat = out_answer_feat.reshape(-1, out_answer_feat.shape[-1])
        ctx_aware_ans_feat = self.ctx2ans_attn(node_feat, answer_feat, out_answer_feat, mask)
        new_node_feat = self.fuse_ctx_ans(torch.cat([node_feat, ctx_aware_ans_feat], -1))

        return new_node_feat


class Context2AnswerAttention(nn.Module):
    def __init__(self, dim, hidden_size):
        super(Context2AnswerAttention, self).__init__()
        self.linear_sim = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, context, answers, out_answers, mask=None):
        """
        Parameters
        :context, (L, dim)
        :answers, (N, dim)
        :out_answers, (N, dim)
        :mask, (L, N)

        Returns
        :ques_emb, (L, dim)
        """
        context_fc = torch.relu(self.linear_sim(context))
        questions_fc = torch.relu(self.linear_sim(answers))

        # shape: (L, N)
        attention = torch.matmul(context_fc, questions_fc.transpose(-1, -2))
        if mask is not None:
            attention = attention.masked_fill_(~mask.bool(), -Constants.INF)
        prob = torch.softmax(attention, dim=-1)
        # shape: (L, dim)
        emb = torch.matmul(prob, out_answers)

        return emb


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.use_copy = self.config['decoder_args']['rnn_decoder_share']['use_copy']
        self.use_coverage = self.config['decoder_args']['rnn_decoder_share']['use_coverage']
        self.logger = Logger(config['out_dir'], config={k:v for k, v in config.items() if k != 'device'}, overwrite=True)
        self.logger.write(config['out_dir'])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        if self.config['graph_construction_args']["graph_construction_share"]["graph_type"] == "dependency":
            topology_builder = DependencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.config['graph_construction_args']["graph_construction_share"]["graph_type"] == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.config['graph_construction_args']["graph_construction_share"]["graph_type"] == "node_emb":
            topology_builder = NodeEmbeddingBasedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_topology_builder = None
        elif self.config['graph_construction_args']["graph_construction_share"]["graph_type"] == "node_emb_refined":
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_graph_type = self.config['graph_construction_args'].graph_construction_private.dynamic_init_graph_type
            if dynamic_init_graph_type is None or dynamic_init_graph_type == 'line':
                dynamic_init_topology_builder = None
            elif dynamic_init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif dynamic_init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            else:
                # dynamic_init_topology_builder
                raise RuntimeError('Define your own dynamic_init_topology_builder')
        else:
            raise NotImplementedError("Define your topology builder.")

        dataset = SQuADDataset(root_dir=self.config['graph_construction_args']['graph_construction_share']['root_dir'],
                              pretrained_word_emb_file=self.config['pre_word_emb_file'],
                              merge_strategy=self.config['graph_construction_args']['graph_construction_private']['merge_strategy'],
                              edge_strategy=self.config['graph_construction_args']["graph_construction_private"]['edge_strategy'],
                              max_word_vocab_size=self.config['top_word_vocab'],
                              min_word_vocab_freq=self.config['min_word_freq'],
                              word_emb_size=self.config['word_emb_size'],
                              share_vocab=self.config['share_vocab'],
                              seed=self.config['seed'],
                              graph_type=graph_type,
                              topology_builder=topology_builder,
                              topology_subdir=self.config['graph_construction_args']['graph_construction_share']['topology_subdir'],
                              dynamic_graph_type=self.config['graph_construction_args']['graph_construction_share']['graph_type'],
                              dynamic_init_topology_builder=dynamic_init_topology_builder,
                              dynamic_init_topology_aux_args={'dummy_param': 0},
                              thread_number=4,
                              port=9000,
                              timeout=15000)

        # TODO: use small ratio of the data (Test only)
        dataset.train = dataset.train[:self.config['n_samples']]
        dataset.val = dataset.val[:self.config['n_samples']]
        dataset.test = dataset.test[:self.config['n_samples']]

        self.train_dataloader = DataLoader(dataset.train, batch_size=self.config['batch_size'], shuffle=True,
                                           num_workers=self.config['num_workers'],
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=self.config['batch_size'], shuffle=False,
                                          num_workers=self.config['num_workers'],
                                          collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.config['batch_size'], shuffle=False,
                                          num_workers=self.config['num_workers'],
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model
        self.num_train = len(dataset.train)
        self.num_val = len(dataset.val)
        self.num_test = len(dataset.test)
        print('Train size: {}, Val size: {}, Test size: {}'
            .format(self.num_train, self.num_val, self.num_test))
        self.logger.write('Train size: {}, Val size: {}, Test size: {}'
            .format(self.num_train, self.num_val, self.num_test))

    def _build_model(self):
        self.model = QGModel(self.vocab, self.config).to(self.config['device'])

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config['lr'])
        self.stopper = EarlyStopping(os.path.join(self.config['out_dir'], Constants._SAVED_WEIGHTS_FILE), patience=self.config['patience'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
            patience=self.config['lr_patience'], verbose=True)

    def _build_evaluation(self):
        self.metrics = {'BLEU': BLEU(n_grams=[1, 2, 3, 4]),
                        'METEOR': METEOR(),
                        'ROUGE': ROUGE()}

    def train(self):
        dur = []
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = []
            t0 = time.time()
            for i, data in enumerate(self.train_dataloader):
                data = all_to_cuda(data, self.config['device'])

                oov_dict = None
                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(data['graph_data'], self.vocab, gt_str=data['tgt_text'])
                    data['tgt_tensor'] = tgt

                logits, loss = self.model(data, oov_dict=oov_dict, require_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.get('grad_clipping', None) not in (None, 0):
                    # Clip gradients
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    # if self.config['use_bert'] and self.config.get('finetune_bert', None):
                    #     parameters += [p for p in self.config['bert_model'].parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

                self.optimizer.step()
                train_loss.append(loss.item())

                pred = torch.max(logits, dim=-1)[1].cpu()
                dur.append(time.time() - t0)

            val_scores = self.evaluate(self.val_dataloader)
            self.scheduler.step(val_scores[self.config['early_stop_metric']])
            format_str = 'Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Val scores:'.format(epoch + 1, self.config['epochs'], np.mean(dur), np.mean(train_loss))
            format_str += self.metric_to_str(val_scores)
            print(format_str)
            self.logger.write(format_str)

            if self.stopper.step(val_scores[self.config['early_stop_metric']], self.model):
                break

        return self.stopper.best_score

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                data = all_to_cuda(data, self.config['device'])

                if self.use_copy:
                    oov_dict = prepare_ext_vocab(data['graph_data'], self.vocab)
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                prob = self.model(data, oov_dict=oov_dict, require_loss=False)
                pred = prob.argmax(dim=-1)

                pred_str = wordid2str(pred.detach().cpu(), ref_dict)
                pred_collect.extend(pred_str)
                gt_collect.extend(data['tgt_text'])

            scores = self.evaluate_predictions(gt_collect, pred_collect)

            return scores

    def translate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                data = all_to_cuda(data, self.config['device'])
                if self.use_copy:
                    oov_dict = prepare_ext_vocab(data['graph_data'], self.vocab)
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                batch_gd = self.model.encode_init_node_feature(data)
                prob = self.model.g2s.encoder_decoder_beam_search(batch_gd, data['graph_data'], self.config['beam_size'], topk=1, oov_dict=oov_dict)

                pred_ids = torch.zeros(len(prob), self.config['decoder_args']['rnn_decoder_private']['max_decoder_step']).fill_(ref_dict.EOS).to(self.config['device']).int()
                for i, item in enumerate(prob):
                    item = item[0]
                    seq = [j.view(1, 1) for j in item]
                    seq = torch.cat(seq, dim=1)
                    pred_ids[i, :seq.shape[1]] = seq

                pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

                pred_collect.extend(pred_str)
                gt_collect.extend(data['tgt_text'])

            scores = self.evaluate_predictions(gt_collect, pred_collect)

            return scores

    def test(self):
        # restored best saved model
        self.stopper.load_checkpoint(self.model)

        t0 = time.time()
        scores = self.translate(self.test_dataloader)
        dur = time.time() - t0
        format_str = 'Test examples: {} | Time: {:.2f}s |  Test scores:'.format(self.num_test, dur)
        format_str += self.metric_to_str(scores)
        print(format_str)
        self.logger.write(format_str)

        return scores

    def evaluate_predictions(self, ground_truth, predict):
        output = {}
        for name, scorer in self.metrics.items():
            score = scorer.calculate_scores(ground_truth=ground_truth, predict=predict)
            if name.upper() == 'BLEU':
                for i in range(len(score[0])):
                    output['BLEU_{}'.format(i + 1)] = score[0][i]
            else:
                output[name] = score[0]

        return output

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' {} = {:0.5f},'.format(k.upper(), metrics[k])

        return format_str[:-1]


def main(config):
    # configure
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    if not config['no_cuda'] and torch.cuda.is_available():
        print('[ Using CUDA ]')
        config['device'] = torch.device('cuda' if config['gpu'] < 0 else 'cuda:%d' % config['gpu'])
        cudnn.benchmark = True
        torch.cuda.manual_seed(config['seed'])
    else:
        config['device'] = torch.device('cpu')

    ts = datetime.datetime.now().timestamp()
    config['out_dir'] += '_{}'.format(ts)
    print('\n' + config['out_dir'])

    runner = ModelHandler(config)
    t0 = time.time()

    val_score = runner.train()
    test_scores = runner.test()

    print('Removed best saved model file to save disk space')
    os.remove(runner.stopper.save_model_path)
    runtime = time.time() - t0
    print('Total runtime: {:.2f}s'.format(time.time() - t0))
    runner.logger.write('Total runtime: {:.2f}s\n'.format(runtime))
    runner.logger.close()

    return val_score, test_scores

def wordid2str(word_ids, vocab):
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

def all_to_cuda(data, device=None):
    if isinstance(data, torch.Tensor):
        data = to_cuda(data, device)
    elif isinstance(data, (list, dict)):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k], device)

    return data

################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_config', '--task_config', required=True, type=str, help='path to the config file')
    parser.add_argument('-g2s_config', '--g2s_config', required=True, type=str, help='path to the config file')
    parser.add_argument('--grid_search', action='store_true', help='flag: grid search')
    args = vars(parser.parse_args())

    return args


def print_config(config):
    print('**************** MODEL CONFIGURATION ****************')
    for key in sorted(config.keys()):
        val = config[key]
        keystr = '{}'.format(key) + (' ' * (24 - len(key)))
        print('{} -->   {}'.format(keystr, val))
    print('**************** MODEL CONFIGURATION ****************')


def grid_search_main(config):
    grid_search_hyperparams = []
    log_path = config['out_dir']
    for k, v in config.items():
        if isinstance(v, list):
            grid_search_hyperparams.append(k)
            log_path += '_{}_{}'.format(k, v)

    logger = Logger(log_path, config=config, overwrite=True)

    best_config = None
    best_score = -1
    best_scores = None
    configs = grid(config)
    for cnf in configs:
        for k in grid_search_hyperparams:
            cnf['out_dir'] += '_{}_{}'.format(k, cnf[k])

        val_score, test_scores = main(cnf)
        if best_score < test_scores[cnf['early_stop_metric']]:
            best_score = test_scores[cnf['early_stop_metric']]
            best_scores = test_scores
            best_config = cnf
            print('Found a better configuration: {}'.format(best_scores))
            logger.write('Found a better configuration: {}'.format(best_scores))

    print('\nBest configuration:')
    logger.write('\nBest configuration:')
    for k in grid_search_hyperparams:
        print('{}: {}'.format(k, best_config[k]))
        logger.write('{}: {}'.format(k, best_config[k]))

    print('Best score: {}'.format(best_scores))
    logger.write('Best score: {}\n'.format(best_scores))
    logger.close()


if __name__ == '__main__':
    cfg = get_args()
    task_args = get_yaml_config(cfg['task_config'])
    g2s_args = get_yaml_config(cfg['g2s_config'])
    # load Graph2Seq template config
    g2s_template = get_basic_args(graph_construction_name=g2s_args['graph_construction_name'],
                              graph_embedding_name=g2s_args['graph_embedding_name'],
                              decoder_name=g2s_args['decoder_name'])
    update_values(to_args=g2s_template, from_args_list=[g2s_args, task_args])

    print_config(g2s_template)
    if cfg['grid_search']:
        grid_search_main(g2s_template)
    else:
        main(g2s_template)
