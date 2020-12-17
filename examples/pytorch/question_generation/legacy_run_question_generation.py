import os
import time
import datetime
import argparse
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
from graph4nlp.pytorch.modules.graph_embedding import GAT, GraphSAGE, GGNN
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.utils.generic_utils import grid, to_cuda, dropout_fn, sparse_mx_to_torch_sparse_tensor, EarlyStopping
from examples.pytorch.semantic_parsing.graph2seq.nouse_loss import Graph2SeqLoss, CoverageLoss
from graph4nlp.pytorch.modules.evaluation import BLEU, METEOR, ROUGE
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils import constants as Constants
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class QGModel(nn.Module):
    def __init__(self, vocab, config):
        super(QGModel, self).__init__()
        self.config = config
        self.vocab = vocab
        embedding_style = {'single_token_item': True if config['graph_type'] != 'ie' else False,
                            'emb_strategy': config.get('emb_strategy', 'w2v_bilstm'),
                            'num_rnn_layers': 1,
                            'bert_model_name': config.get('bert_model_name', 'bert-base-uncased'),
                            'bert_lower_case': True
                           }
        print('embedding_style: {}'.format(embedding_style))

        assert not (config['graph_type'] in ('node_emb', 'node_emb_refined') and config['gnn'] == 'gat'), \
                                'dynamic graph construction does not support GAT'

        use_edge_weight = False
        if config['graph_type'] == 'dependency':
            self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=config['num_hidden'],
                                                                   word_dropout=config['word_dropout'],
                                                                   rnn_dropout=config['enc_rnn_dropout'],
                                                                   fix_word_emb=not config['no_fix_word_emb'],
                                                                   fix_bert_emb=not config.get('no_fix_bert_emb', False),
                                                                   device=config['device'])
        elif config['graph_type'] == 'constituency':
            self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=config['num_hidden'],
                                                                   word_dropout=config['word_dropout'],
                                                                   rnn_dropout=config['enc_rnn_dropout'],
                                                                   fix_word_emb=not config['no_fix_word_emb'],
                                                                   device=config['device'])
        elif config['graph_type'] == 'ie':
            self.graph_topology = IEBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=config['num_hidden'],
                                                                   word_dropout=config['word_dropout'],
                                                                   rnn_dropout=config['enc_rnn_dropout'],
                                                                   fix_word_emb=not config['no_fix_word_emb'],
                                                                   device=config['device'])
        elif config['graph_type'] == 'node_emb':
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                                    vocab.in_word_vocab,
                                    embedding_style,
                                    sim_metric_type=config['gl_metric_type'],
                                    num_heads=config['gl_num_heads'],
                                    top_k_neigh=config['gl_top_k'],
                                    epsilon_neigh=config['gl_epsilon'],
                                    smoothness_ratio=config['gl_smoothness_ratio'],
                                    connectivity_ratio=config['gl_connectivity_ratio'],
                                    sparsity_ratio=config['gl_sparsity_ratio'],
                                    input_size=config['num_hidden'],
                                    hidden_size=config['gl_num_hidden'],
                                    fix_word_emb=not config['no_fix_word_emb'],
                                    word_dropout=config['word_dropout'],
                                    rnn_dropout=config['enc_rnn_dropout'],
                                    device=config['device'])
            use_edge_weight = True
        elif config['graph_type'] == 'node_emb_refined':
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                                    vocab.in_word_vocab,
                                    embedding_style,
                                    config['init_adj_alpha'],
                                    sim_metric_type=config['gl_metric_type'],
                                    num_heads=config['gl_num_heads'],
                                    top_k_neigh=config['gl_top_k'],
                                    epsilon_neigh=config['gl_epsilon'],
                                    smoothness_ratio=config['gl_smoothness_ratio'],
                                    connectivity_ratio=config['gl_connectivity_ratio'],
                                    sparsity_ratio=config['gl_sparsity_ratio'],
                                    input_size=config['num_hidden'],
                                    hidden_size=config['gl_num_hidden'],
                                    fix_word_emb=not config['no_fix_word_emb'],
                                    word_dropout=config['word_dropout'],
                                    rnn_dropout=config['enc_rnn_dropout'],
                                    device=config['device'])
            use_edge_weight = True
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(config['graph_type']))


        if 'w2v' in self.graph_topology.embedding_layer.word_emb_layers:
            self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                            self.vocab.in_word_vocab.embeddings.shape[0],
                            self.vocab.in_word_vocab.embeddings.shape[1],
                            pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                            fix_emb=not config['no_fix_word_emb'],
                            device=config['device']).word_emb_layer

        answer_feat_size = self.vocab.in_word_vocab.embeddings.shape[1]
        if 'node_edge_bert' in self.graph_topology.embedding_layer.word_emb_layers:
            self.bert_encoder = self.graph_topology.embedding_layer.word_emb_layers['node_edge_bert']
            answer_feat_size += self.bert_encoder.bert_model.config.hidden_size
        elif 'seq_bert' in self.graph_topology.embedding_layer.word_emb_layers:
            self.bert_encoder = self.graph_topology.embedding_layer.word_emb_layers['seq_bert']
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


        if config['gnn'] == 'gat':
            heads = [config['gat_num_heads']] * (config['gnn_num_layers'] - 1) + [config['gat_num_out_heads']]
            self.gnn = GAT(config['gnn_num_layers'],
                        config['num_hidden'],
                        config['num_hidden'],
                        config['num_hidden'],
                        heads,
                        direction_option=config['gnn_direction_option'],
                        feat_drop=config['gnn_dropout'],
                        attn_drop=config['gat_attn_dropout'],
                        negative_slope=config['gat_negative_slope'],
                        residual=config['gat_residual'],
                        activation=F.elu)
        elif config['gnn'] == 'graphsage':
            self.gnn = GraphSAGE(config['gnn_num_layers'],
                        config['num_hidden'],
                        config['num_hidden'],
                        config['num_hidden'],
                        config['graphsage_aggreagte_type'],
                        direction_option=config['gnn_direction_option'],
                        feat_drop=config['gnn_dropout'],
                        bias=True,
                        norm=None,
                        activation=F.relu,
                        use_edge_weight=use_edge_weight)
        elif config['gnn'] == 'ggnn':
            self.gnn = GGNN(config['gnn_num_layers'],
                        config['num_hidden'],
                        config['num_hidden'],
                        config['num_hidden'],
                        feat_drop=config['gnn_dropout'],
                        direction_option=config['gnn_direction_option'],
                        bias=True,
                        use_edge_weight=use_edge_weight)
        else:
            raise RuntimeError('Unknown gnn type: {}'.format(config['gnn']))

        self.seq_decoder = StdRNNDecoder(config['max_dec_steps'],
                                         2 * config['num_hidden'] if config['gnn_direction_option'] == 'bi_sep' else config['num_hidden'],
                                         config['num_hidden'],
                                         self.word_emb,
                                         self.vocab.in_word_vocab,
                                         graph_pooling_strategy=config['graph_pooling_strategy'],
                                         attention_type=config['dec_attention_type'],
                                         rnn_emb_input_size=config['num_hidden'],
                                         node_type_num=None,
                                         fuse_strategy=config['dec_fuse_strategy'],
                                         use_copy=config.get('use_copy', False),
                                         use_coverage=config['use_coverage'],
                                         coverage_strategy=config.get('coverage_strategy', 'sum'),
                                         tgt_emb_as_output_layer=config['tgt_emb_as_output_layer'],
                                         dropout=config['dec_rnn_dropout'])
        self.loss_calc = Graph2SeqLoss(self.vocab.in_word_vocab)
        self.loss_cover = CoverageLoss(config['coverage_loss_ratio'])


    def forward(self, data, require_loss=True):
        num_graph_nodes = []
        for g in data['graph_data']:
            num_graph_nodes.append(g.get_node_num())

        # graph embedding construction
        batch_gd = self.graph_topology(data['graph_data'])

        # answer alignment
        answer_feat = self.word_emb(data['input_tensor2'])
        answer_feat = dropout_fn(answer_feat, self.config['word_dropout'], shared_axes=[-2], training=self.training)

        if self.bert_encoder is not None:
            answer_bert_feat = self.bert_encoder(data['input_text2'])
            answer_feat = torch.cat([answer_feat, answer_bert_feat], -1)

        answer_feat = self.ans_rnn_encoder(answer_feat, data['input_length2'])[0]
        new_node_feat = self.answer_alignment(batch_gd.node_features['node_feat'], answer_feat, answer_feat, num_graph_nodes, data['input_length2'])
        batch_gd.node_features['node_feat'] = new_node_feat

        # run GNN
        self.gnn(batch_gd)
        batch_gd.node_features['rnn_emb'] = batch_gd.node_features['node_feat']

        # seq decoder
        prob, enc_attn_weights, coverage_vectors = self.seq_decoder(from_batch(batch_gd), tgt_seq=data['tgt_tensor'])
        if require_loss:
            loss = self.loss_calc(prob, data['tgt_tensor'])
            loss += self.loss_cover(enc_attn_weights, coverage_vectors)
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
        self.logger = Logger(config['out_dir'], config={k:v for k, v in config.items() if k != 'device'}, overwrite=True)
        self.logger.write(config['out_dir'])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        dynamic_init_topology_builder = None
        if self.config['graph_type'] == 'dependency':
            topology_builder = DependencyBasedGraphConstruction
            graph_type = 'static'
            merge_strategy = 'tailhead'
        elif self.config['graph_type'] == 'constituency':
            topology_builder = ConstituencyBasedGraphConstruction
            graph_type = 'static'
            merge_strategy = 'tailhead'
        elif self.config['graph_type'] == 'ie':
            topology_builder = IEBasedGraphConstruction
            graph_type = 'static'
            merge_strategy = 'global'
        elif self.config['graph_type'] == 'node_emb':
            topology_builder = NodeEmbeddingBasedGraphConstruction
            graph_type = 'dynamic'
            merge_strategy = None
        elif self.config['graph_type'] == 'node_emb_refined':
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            graph_type = 'dynamic'
            merge_strategy = 'tailhead'

            if self.config['init_graph_type'] == 'line':
                dynamic_init_topology_builder = None
            elif self.config['init_graph_type'] == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif self.config['init_graph_type'] == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            elif self.config['init_graph_type'] == 'ie':
                merge_strategy = 'global'
                dynamic_init_topology_builder = IEBasedGraphConstruction
            else:
                # dynamic_init_topology_builder
                raise RuntimeError('Define your own dynamic_init_topology_builder')
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(self.config['graph_type']))

        topology_subdir = '{}_based_graph'.format(self.config['graph_type'])
        if self.config['graph_type'] == 'node_emb_refined':
            topology_subdir += '_{}'.format(self.config['init_graph_type'])

        dataset = SQuADDataset(root_dir=self.config.get('root_dir', 'examples/pytorch/question_generation/data/squad_split2'),
                              pretrained_word_emb_file=self.config['pre_word_emb_file'],
                              merge_strategy=merge_strategy,
                              seed=self.config['seed'],
                              graph_type=graph_type,
                              topology_builder=topology_builder,
                              topology_subdir=topology_subdir,
                              dynamic_graph_type=self.config['graph_type'] if self.config['graph_type'] in ('node_emb', 'node_emb_refined') else None,
                              dynamic_init_topology_builder=dynamic_init_topology_builder,
                              dynamic_init_topology_aux_args={'dummy_param': 0})

        # TODO: use small ratio of the data
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
                logits, loss = self.model(data, require_loss=True)
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
                prob = self.model(data, require_loss=False)
                pred = prob.argmax(dim=-1)

                pred_str = wordid2str(pred.detach().cpu(), self.vocab.in_word_vocab)
                tgt_str = wordid2str(data['tgt_tensor'], self.vocab.in_word_vocab)
                pred_collect.extend(pred_str)
                gt_collect.extend(data['tgt_text'])

            scores = self.evaluate_predictions(gt_collect, pred_collect)

            return scores

    def test(self):
        # restored best saved model
        self.stopper.load_checkpoint(self.model)

        t0 = time.time()
        scores = self.evaluate(self.test_dataloader)
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
        ret.append(' '.join(ret_inst))

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
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--grid_search', action='store_true', help='flag: grid search')
    args = vars(parser.parse_args())

    return args


def get_config(config_path='config.yml'):
    with open(config_path, 'r') as setting:
        config = yaml.load(setting)

    return config


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
    config = get_config(cfg['config'])
    print_config(config)
    if cfg['grid_search']:
        grid_search_main(config)
    else:
        main(config)
