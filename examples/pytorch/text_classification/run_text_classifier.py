import os
import time
import datetime
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

from graph4nlp.pytorch.datasets.trec import TrecDataset
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import WordEmbedding
from graph4nlp.pytorch.modules.graph_embedding import GAT, GraphSAGE, GGNN
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN
from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.utils.generic_utils import grid, to_cuda, EarlyStopping
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils import constants as Constants
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



class TextClassifier(nn.Module):
    def __init__(self, vocab, config):
        super(TextClassifier, self).__init__()
        self.config = config
        self.vocab = vocab
        embedding_style = {'single_token_item': True if config['graph_type'] != 'ie' else False,
                            'emb_strategy': config.get('emb_strategy', 'w2v_bilstm'),
                            'num_rnn_layers': 1,
                            'bert_model_name': config.get('bert_model_name', 'bert-base-uncased'),
                            'bert_lower_case': True
                           }

        assert not (config['graph_type'] in ('node_emb', 'node_emb_refined') and config['gnn'] == 'gat'), \
                                'dynamic graph construction does not support GAT'

        use_edge_weight = False
        if config['graph_type'] == 'dependency':
            self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=config['num_hidden'],
                                                                   word_dropout=config['word_dropout'],
                                                                   rnn_dropout=config['rnn_dropout'],
                                                                   fix_word_emb=not config['no_fix_word_emb'],
                                                                   fix_bert_emb=not config.get('no_fix_bert_emb', False),
                                                                   device=config['device'])
        elif config['graph_type'] == 'constituency':
            self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=config['num_hidden'],
                                                                   word_dropout=config['word_dropout'],
                                                                   rnn_dropout=config['rnn_dropout'],
                                                                   fix_word_emb=not config['no_fix_word_emb'],
                                                                   fix_bert_emb=not config.get('no_fix_bert_emb', False),
                                                                   device=config['device'])
        elif config['graph_type'] == 'ie':
            self.graph_topology = IEBasedGraphConstruction(embedding_style=embedding_style,
                                                                   vocab=vocab.in_word_vocab,
                                                                   hidden_size=config['num_hidden'],
                                                                   word_dropout=config['word_dropout'],
                                                                   rnn_dropout=config['rnn_dropout'],
                                                                   fix_word_emb=not config['no_fix_word_emb'],
                                                                   fix_bert_emb=not config.get('no_fix_bert_emb', False),
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
                                    fix_bert_emb=not config.get('no_fix_bert_emb', False),
                                    word_dropout=config['word_dropout'],
                                    rnn_dropout=config['rnn_dropout'],
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
                                    fix_bert_emb=not config.get('no_fix_bert_emb', False),
                                    word_dropout=config['word_dropout'],
                                    rnn_dropout=config['rnn_dropout'],
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

        self.clf = FeedForwardNN(2 * config['num_hidden'] if config['gnn_direction_option'] == 'bi_sep' else config['num_hidden'],
                        config['num_classes'],
                        [config['num_hidden']],
                        graph_pool_type=config['graph_pooling'],
                        dim=config['num_hidden'],
                        use_linear_proj=config['max_pool_linear_proj'])

        self.loss = GeneralLoss('CrossEntropy')


    def forward(self, graph_list, tgt=None, require_loss=True):
        # graph embedding construction
        batch_gd = self.graph_topology(graph_list)

        # run GNN
        self.gnn(batch_gd)

        # run graph classifier
        self.clf(batch_gd)
        logits = batch_gd.graph_attributes['logits']

        if require_loss:
            loss = self.loss(logits, tgt)
            return logits, loss
        else:
            return logits


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.logger = Logger(self.config['out_dir'], config={k:v for k, v in self.config.items() if k != 'device'}, overwrite=True)
        self.logger.write(self.config['out_dir'])
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

        dataset = TrecDataset(root_dir=self.config.get('root_dir', 'examples/pytorch/text_classification/data/trec'),
                              pretrained_word_emb_file=self.config['pre_word_emb_file'],
                              merge_strategy=merge_strategy,
                              seed=self.config['seed'],
                              device=config['device'],
                              thread_number=4,
                              port=9000,
                              timeout=15000,
                              word_emb_size=300,
                              graph_type=graph_type,
                              topology_builder=topology_builder,
                              topology_subdir=topology_subdir,
                              dynamic_graph_type=self.config['graph_type'] if self.config['graph_type'] in ('node_emb', 'node_emb_refined') else None,
                              dynamic_init_topology_builder=dynamic_init_topology_builder,
                              dynamic_init_topology_aux_args={'dummy_param': 0},
                              thread_number=4,
                              port=9000,
                              timeout=15000)

        self.train_dataloader = DataLoader(dataset.train, batch_size=self.config['batch_size'], shuffle=True,
                                           num_workers=self.config['num_workers'],
                                           collate_fn=dataset.collate_fn)
        if hasattr(dataset, 'val')==False:
            dataset.val = dataset.test
        self.val_dataloader = DataLoader(dataset.val, batch_size=self.config['batch_size'], shuffle=False,
                                          num_workers=self.config['num_workers'],
                                          collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.config['batch_size'], shuffle=False,
                                          num_workers=self.config['num_workers'],
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model
        self.config['num_classes'] = dataset.num_classes
        self.num_train = len(dataset.train)
        self.num_val = len(dataset.val)
        self.num_test = len(dataset.test)
        print('Train size: {}, Val size: {}, Test size: {}'
            .format(self.num_train, self.num_val, self.num_test))
        self.logger.write('Train size: {}, Val size: {}, Test size: {}'
            .format(self.num_train, self.num_val, self.num_test))

    def _build_model(self):
        self.model = TextClassifier(self.vocab, self.config).to(self.config['device'])

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config['lr'])
        self.stopper = EarlyStopping(os.path.join(self.config['out_dir'], Constants._SAVED_WEIGHTS_FILE), patience=self.config['patience'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
            patience=self.config['lr_patience'], verbose=True)

    def _build_evaluation(self):
        self.metric = Accuracy(['accuracy'])

    def train(self):
        dur = []
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = []
            train_acc = []
            t0 = time.time()
            for i, data in enumerate(self.train_dataloader):
                graph_list, tgt = data
                graph_list = [to_cuda(g, self.config['device']) for g in graph_list]
                tgt = to_cuda(tgt, self.config['device'])
                logits, loss = self.model(graph_list, tgt, require_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                pred = torch.max(logits, dim=-1)[1].cpu()
                train_acc.append(self.metric.calculate_scores(ground_truth=tgt.cpu(), predict=pred.cpu())[0])
                dur.append(time.time() - t0)

            val_acc = self.evaluate(self.val_dataloader)
            self.scheduler.step(val_acc)
            print('Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}'.
              format(epoch + 1, self.config['epochs'], np.mean(dur), np.mean(train_loss), np.mean(train_acc), val_acc))
            self.logger.write('Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}'.
                        format(epoch + 1, self.config['epochs'], np.mean(dur), np.mean(train_loss), np.mean(train_acc), val_acc))

            if self.stopper.step(val_acc, self.model):
                break

        return self.stopper.best_score

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                graph_list, tgt = data
                graph_list = [to_cuda(g, self.config['device']) for g in graph_list]
                logits = self.model(graph_list, require_loss=False)
                pred_collect.append(logits)
                gt_collect.append(tgt)

            pred_collect = torch.max(torch.cat(pred_collect, 0), dim=-1)[1].cpu()
            gt_collect = torch.cat(gt_collect, 0).cpu()
            score = self.metric.calculate_scores(ground_truth=gt_collect, predict=pred_collect)[0]

            return score

    def test(self):
        # restored best saved model
        self.stopper.load_checkpoint(self.model)

        t0 = time.time()
        acc = self.evaluate(self.test_dataloader)
        dur = time.time() - t0
        print('Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}'.
          format(self.num_test, dur, acc))
        self.logger.write('Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}'.
          format(self.num_test, dur, acc))

        return acc


def main(config):
    # configure
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    if not config['no_cuda'] and torch.cuda.is_available():
        print('[ Using CUDA ]')
        config['device'] = torch.device('cuda' if config['gpu'] < 0 else 'cuda:%d' % config['gpu'])
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        config['device'] = torch.device('cpu')

    ts = datetime.datetime.now().timestamp()
    config['out_dir'] += '_{}'.format(ts)
    print('\n' + config['out_dir'])

    runner = ModelHandler(config)
    t0 = time.time()

    val_acc = runner.train()
    test_acc = runner.test()

    print('Removed best saved model file to save disk space')
    os.remove(runner.stopper.save_model_path)
    runtime = time.time() - t0
    print('Total runtime: {:.2f}s'.format(runtime))
    runner.logger.write('Total runtime: {:.2f}s\n'.format(runtime))
    runner.logger.close()

    return val_acc, test_acc


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
    configs = grid(config)
    for cnf in configs:
        for k in grid_search_hyperparams:
            cnf['out_dir'] += '_{}_{}'.format(k, cnf[k])

        val_score, test_score = main(cnf)
        if best_score < test_score:
            best_score = test_score
            best_config = cnf
            print('Found a better configuration: {}'.format(best_score))
            logger.write('Found a better configuration: {}'.format(best_score))

    print('\nBest configuration:')
    logger.write('\nBest configuration:')
    for k in grid_search_hyperparams:
        print('{}: {}'.format(k, best_config[k]))
        logger.write('{}: {}'.format(k, best_config[k]))

    print('Best score: {}'.format(best_score))
    logger.write('Best score: {}\n'.format(best_score))
    logger.close()


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    print_config(config)
    if cfg['grid_search']:
        grid_search_main(config)
    else:
        main(config)

# screen -S ghn_code3  python -m  examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_undirected_constituency.yaml
# screen -S ghn_code1  python -m  examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/graphsage_undirected_node_emb.yaml
# screen -S ghn_code1  python -m  examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_undirected_constituency.yaml
# screen -S ghn_code3 python -m  examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_dependency.yaml
# screen -S ghn_code1 python -m  examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_undirected_dependency.yaml --grid_search
# screen -S ghn_code1 python -m  examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_bi_sep_dependency.yaml --grid_search
# screen -S ghn_code1 python -m  examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/gat_bi_sep_dependency.yaml --grid_search
