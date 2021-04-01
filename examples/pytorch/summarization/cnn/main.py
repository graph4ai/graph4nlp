import os
import datetime
import copy
import torch
import time
import torch.backends.cudnn as cudnn
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from .dataset import CNNDataset
from .model_g2s import Graph2seq
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
from graph4nlp.pytorch.modules.utils.generic_utils import grid, to_cuda, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader
import torch.optim as optim
from .config_g2s import get_args
from .utils import get_log, wordid2str
from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
from graph4nlp.pytorch.data.data import from_batch, GraphData
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.evaluation import BLEU
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab

def all_to_cuda(data, device=None):
    if isinstance(data, torch.Tensor):
        data = to_cuda(data, device)
    elif isinstance(data, (list, dict)):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k], device)

    return data

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
        if 'DependencyGraph' in self.opt.topology_subdir:
            graph_type = 'static'
            topology_builder = DependencyBasedGraphConstruction
            topology_subdir = self.opt.topology_subdir
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
                             tokenizer=None,
                             device=self.device,
                             word_emb_size=self.opt.word_emb_size,
                             thread_number=35,
                             share_vocab=True,
                             graph_type=graph_type,
                             topology_builder=topology_builder,
                             topology_subdir=topology_subdir,
                             dynamic_graph_type=dynamic_graph_type,
                             dynamic_init_topology_builder=dynamic_init_topology_builder,
                             dynamic_init_topology_aux_args={'dummy_param': 0})

        self.train_dataloader = DataLoader(dataset.train, batch_size=self.opt.batch_size, shuffle=True, num_workers=0,
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=self.opt.batch_size, shuffle=False, num_workers=0,
                                         collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.opt.batch_size, shuffle=False, num_workers=0,
                                          collate_fn=dataset.collate_fn)
        self.vocab: VocabModel = dataset.vocab_model

    def _build_model(self):
        self.model = Graph2seq(self.vocab,
                               use_copy=self.opt.use_copy,
                               use_coverage=self.opt.use_coverage,
                               gnn=self.opt.gnn,
                               device=self.device,
                               rnn_dropout=self.opt.rnn_dropout,
                               emb_dropout=self.opt.word_dropout,
                               hidden_size=self.opt.hidden_size).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt.learning_rate)

    def _build_evaluation(self):
        self.metrics = [ROUGE()]

    def train(self):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(200):
            # score = self.evaluate(split="val")
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

    def prepare_ext_vocab(self, batch, vocab, gt_str=None):
        oov_dict = copy.deepcopy(vocab.in_word_vocab)
        for g in batch:
            token_matrix = []
            for node_idx in range(g.get_node_num()):
                node_token = g.node_attributes[node_idx]['token']
                if oov_dict.getIndex(node_token) == oov_dict.UNK:
                    oov_dict._add_words([node_token])
                token_matrix.append([oov_dict.getIndex(node_token)])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long).to(self.device)
            g.node_features['token_id_oov'] = token_matrix

        if gt_str is not None:
            oov_tgt_collect = []
            for s in gt_str:
                oov_tgt = oov_dict.to_index_sequence(s)
                oov_tgt.append(oov_dict.EOS)
                oov_tgt = np.array(oov_tgt)
                oov_tgt_collect.append(oov_tgt)

            output_pad = pad_2d_vals_no_size(oov_tgt_collect)

            tgt_seq = torch.from_numpy(output_pad).long().to(self.device)
            return oov_dict, tgt_seq
        else:
            return oov_dict

    def train_epoch(self, epoch, split="train"):
        assert split in ["train"]
        self.logger.info("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_dataloader
        step_all_train = len(dataloader)
        start = time.time()
        for step, data in enumerate(dataloader):
            graph_list, tgt, gt_str = data
            tgt = tgt.to(self.device)
            oov_dict = None
            if self.opt.use_copy:
                oov_dict, tgt = self.prepare_ext_vocab(graph_list, self.vocab, gt_str=gt_str)

            _, loss = self.model(graph_list, tgt, oov_dict=oov_dict, require_loss=True)
            loss_collect.append(loss.item())
            if step % self.opt.loss_display_step == 0 and step != 0:
                self.logger.info("Epoch {}: [{} / {}] loss: {:.3f}".format(epoch, step, step_all_train,
                                                                           np.mean(loss_collect)))
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, split="val", test_mode=False):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            graph_list, tgt, gt_str = data
            if self.opt.use_copy:
                oov_dict = self.prepare_ext_vocab(graph_list, self.vocab)
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            prob = self.model(graph_list, oov_dict=oov_dict, require_loss=False)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), ref_dict)
            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        if test_mode==True:
            with open(self.opt.checkpoint_save_path+'/cnn_pred_output.txt','w+') as f:
                for line in pred_collect:
                    f.write(line+'\n')

            with open(self.opt.checkpoint_save_path+'/cnn_tgt_output.txt','w+') as f:
                for line in gt_collect:
                    f.write(line+'\n')

        score, _ = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation ROUGE in `{}` split: {:.3f}".format(split, score))
        return score

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_path)

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
        elif self.config['graph_construction_args']["graph_construction_share"]["graph_type"] == "triples":
            topology_builder = None
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
                raise RuntimeError('Define your own dynamic_init_topology_builder')
        else:
            raise NotImplementedError("Define your topology builder.")

        dataset = CNNDataset(root_dir=self.config['graph_construction_args']['graph_construction_share']['root_dir'],
                             pretrained_word_emb_file=self.config['pre_word_emb_file'],
                             merge_strategy=self.config['graph_construction_args']['graph_construction_private'][
                                 'merge_strategy'],
                             edge_strategy=self.config['graph_construction_args']["graph_construction_private"][
                                 'edge_strategy'],
                             max_word_vocab_size=self.config['top_word_vocab'],
                             min_word_vocab_freq=self.config['min_word_freq'],
                             word_emb_size=self.config['word_emb_size'],
                             share_vocab=self.config['share_vocab'],
                             lower_case=self.config['vocab_lower_case'],
                             seed=self.config['seed'],
                             graph_type=graph_type,
                             topology_builder=topology_builder,
                             topology_subdir=self.config['graph_construction_args']['graph_construction_share'][
                                 'topology_subdir'],
                             dynamic_graph_type=self.config['graph_construction_args']['graph_construction_share'][
                                 'graph_type'],
                             dynamic_init_topology_builder=dynamic_init_topology_builder,
                             dynamic_init_topology_aux_args={'dummy_param': 0},
                             thread_number=35,
                             port=9000,
                             timeout=15000,
                             tokenizer=None
                             )

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
        self.model = Graph2Seq.from_args(self.config, self.vocab, self.config['device'])
        self.logger.write(str(self.model))

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config['lr'])
        self.stopper = EarlyStopping(os.path.join(self.config['out_dir'], Constants._SAVED_WEIGHTS_FILE), patience=self.config['patience'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
            patience=self.config['lr_patience'], verbose=True)

    def _build_evaluation(self):
        # self.metrics = {'BLEU': BLEU(n_grams=[1, 2, 3, 4]),
        #                 'METEOR': METEOR(),
        #                 'ROUGE': ROUGE()}
        self.metrics = {'BLEU': BLEU(n_grams=[1, 2, 3, 4])}

    def train(self):
        dur = []
        for epoch in range(self.config['epochs']):
            # val_scores = self.evaluate(self.val_dataloader)
            self.model.train()
            train_loss = []
            t0 = time.time()

            for i, data in enumerate(self.train_dataloader):
                data = all_to_cuda(data, self.config['device'])
                to_cuda(data['graph_data'], self.config['device'])

                oov_dict = None
                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(data['graph_data'],
                                                      self.vocab,
                                                      gt_str=data['output_str'],
                                                      device=self.config['device'])
                    data['tgt_seq'] = tgt

                logits, loss = self.model(data, oov_dict=oov_dict, require_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.get('grad_clipping', None) not in (None, 0):
                    # Clip gradients
                    parameters = [p for p in self.model.parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

                self.optimizer.step()
                train_loss.append(loss.item())
                print('Epoch = {}, Step = {}, Loss = {:.3f}'.format(epoch, i, loss.item()))

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

    def evaluate(self, dataloader, write2file=False, part='dev'):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                data = all_to_cuda(data, self.config['device'])
                to_cuda(data['graph_data'], self.config['device'])

                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(data['graph_data'],
                                                      self.vocab,
                                                      gt_str=data['output_str'],
                                                      device=self.config['device'])
                    data['tgt_text'] = tgt
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                prob = self.model(data, oov_dict=oov_dict, require_loss=False)
                pred = prob.argmax(dim=-1)

                pred_str = wordid2str(pred.detach().cpu(), ref_dict)
                pred_collect.extend(pred_str)
                gt_collect.extend(data['output_str'])

            if write2file == True:
                with open('{}/{}_ent_pred.txt'.format(self.config['out_dir'], self.config['out_dir'].split('/')[-1]), 'w+') as f:
                    for line in pred_collect:
                        f.write(line + '\n')

                pred_file = open('{}/{}_pred.txt'.format(self.config['out_dir'], self.config['out_dir'].split('/')[-1]), 'w+')

            rplc_list = [x.rplc_dict for x in dataloader.dataset]
            assert len(rplc_list) == len(pred_collect)
            rplc_pred_collect = []
            for line, rplc_dict in zip(pred_collect, rplc_list):
                for k, v in rplc_dict.items():
                    k = k.lower()
                    v = v.lower()
                    line = line.replace(k, v)
                rplc_pred_collect.append(line)
                if write2file == True:
                    pred_file.write(line + '\n')

            hyps = relexicalise(rplc_pred_collect, self.config['dataset'], part=part, lowercased=True)

            all_ref_lists = []
            root_dir = 'examples/pytorch/rdf2text/data/{}'.format(self.config['dataset'])
            for i in range(5):
                all_ref_lists.append(open('{}/raw/{}-all-notdelex-reference{}.lex'.format(root_dir, part, i)).readlines())

            all_refs = []
            for i in range(len(all_ref_lists[0])):
                one_refs = []
                for j in range(5):
                    if all_ref_lists[j][i] != '\n':
                        one_refs.append(all_ref_lists[j][i])
                all_refs.append(one_refs)

            scores = self.evaluate_predictions(all_refs, hyps)
            if write2file == True:
                print('BLEU1 = {}, BLEU2 = {}, BLEU3 = {}, BLEU4 = {}'.format(scores['BLEU_1'],
                                                                               scores['BLEU_2'],
                                                                               scores['BLEU_3'],
                                                                               scores['BLEU_4']))

            return scores

    def translate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                print(i)
                data = all_to_cuda(data, self.config['device'])
                if self.use_copy:
                    oov_dict = prepare_ext_vocab(data['graph_data'], self.vocab, device=self.config['device'])
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                batch_gd = self.model.encode_init_node_feature(data)
                prob = self.model.g2s.encoder_decoder_beam_search(batch_gd,
                                                                  data['graph_data'],
                                                                  data['gmp_seq'],
                                                                  data['gmp_jump'],
                                                                  self.config['beam_size'],
                                                                  topk=1,
                                                                  oov_dict=oov_dict)

                pred_ids = torch.zeros(len(prob), self.config['decoder_args']['rnn_decoder_private']['max_decoder_step']).fill_(ref_dict.EOS).to(self.config['device']).int()
                for i, item in enumerate(prob):
                    item = item[0]
                    seq = [j.view(1, 1) for j in item]
                    seq = torch.cat(seq, dim=1)
                    pred_ids[i, :seq.shape[1]] = seq

                pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

                pred_collect.extend(pred_str)
                gt_collect.extend(data['output_str'])

            # scores = self.evaluate_predictions(gt_collect, pred_collect)

            with open('{}/{}_ent_pred.txt'.format(self.config['out_dir'], self.config['out_dir'].split('/')[-1]), 'w+') as f:
                for line in pred_collect:
                    f.write(line + '\n')

            rplc_list = [x.rplc_dict for x in dataloader.dataset]
            assert len(rplc_list) == len(pred_collect)
            rplc_pred_collect = []
            with open('{}/{}_pred.txt'.format(self.config['out_dir'], self.config['out_dir'].split('/')[-1]), 'w+') as f:
                for line, rplc_dict in zip(pred_collect, rplc_list):
                    for k, v in rplc_dict.items():
                        k = k.lower()
                        v = v.lower()
                        line = line.replace(k, v)
                    f.write(line + '\n')
                    rplc_pred_collect.append(line)

            hyps = relexicalise(rplc_pred_collect, self.config['dataset'], part='test', lowercased=True)

            all_ref_lists = []

            root_dir = 'examples/pytorch/rdf2text/data/{}'.format(self.config['dataset'])

            for i in range(5):
                all_ref_lists.append(open('{}/raw/test-all-notdelex-reference{}.lex'.format(root_dir, i)).readlines())

            all_refs = []
            for i in range(len(all_ref_lists[0])):
                one_refs = []
                for j in range(5):
                    if all_ref_lists[j][i] != '\n':
                        one_refs.append(all_ref_lists[j][i])
                all_refs.append(one_refs)

            scores = self.evaluate_predictions(all_refs, hyps)

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

def print_config(config):
    print('**************** MODEL CONFIGURATION ****************')
    for key in sorted(config.keys()):
        val = config[key]
        keystr = '{}'.format(key) + (' ' * (24 - len(key)))
        print('{} -->   {}'.format(keystr, val))
    print('**************** MODEL CONFIGURATION ****************')

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
    # config['out_dir'] += '_{}'.format(ts)
    print('\n' + config['out_dir'])

    runner = ModelHandler(config)
    t0 = time.time()

    val_score = runner.train()
    # test_scores = runner.test()
    runner.stopper.load_checkpoint(runner.model)
    test_scores = runner.evaluate(runner.test_dataloader, write2file=True, part='test')

    # print('Removed best saved model file to save disk space')
    # os.remove(runner.stopper.save_model_path)
    runtime = time.time() - t0
    print('Total runtime: {:.2f}s'.format(time.time() - t0))
    runner.logger.write('Total runtime: {:.2f}s\n'.format(runtime))
    runner.logger.close()

    return val_score, test_scores

if __name__ == "__main__":
    cfg = get_args()
    task_args = get_yaml_config(cfg['task_config'])
    g2s_args = get_yaml_config(cfg['g2s_config'])
    # load Graph2Seq template config
    g2s_template = get_basic_args(graph_construction_name=g2s_args['graph_construction_name'],
                                  graph_embedding_name=g2s_args['graph_embedding_name'],
                                  decoder_name=g2s_args['decoder_name'])
    update_values(to_args=g2s_template, from_args_list=[g2s_args, task_args])
    print_config(g2s_template)
    main(g2s_template)

    # runner = CNN(opt)
    # max_score = runner.train()
    # print("Train finish, best val score: {:.3f}".format(max_score))
    # runner.load_checkpoint('best.pth')
    # runner.evaluate(split="test", test_mode=True)