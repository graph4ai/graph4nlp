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

from graph4nlp.pytorch.datasets.webnlg_gmp import WebNLGGMPDataset
from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import RNNEmbedding, WordEmbedding
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.models.graphgmp2seq import GraphGMP2Seq
from graph4nlp.pytorch.models.gmp2seq import GMP2Seq
from graph4nlp.pytorch.modules.utils.generic_utils import grid, to_cuda, dropout_fn, sparse_mx_to_torch_sparse_tensor, EarlyStopping
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.models.graphgmp2seq_loss import GraphGMP2SeqLoss
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.evaluation import BLEU, METEOR, ROUGE
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import DecoderStrategy
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config
# import multiprocessing
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# multiprocessing.set_start_method("spawn", force=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def relexicalise(relex_predictions, dataset_name, part='dev', lowercased=True):
    """
    Take a file from seq2seq output and write a relexicalised version of it.
    :param rplc_list: list of dictionaries of replacements for each example (UPPER:not delex item)
    :return: list of predicted sentences
    """

    # create a mapping between not delex triples and relexicalised sents
    with open('examples/pytorch/rdf2text/data/{}/raw/'.format(dataset_name)+part+'-webnlg-all-notdelex.triple', 'r') as f:
        dev_sources = [line.strip() for line in f]
    src_gens = {}
    for src, gen in zip(dev_sources, relex_predictions):
        src_gens[src] = gen  # need only one lex, because they are the same for a given triple

    # write generated sents to a file in the same order as triples are written in the source file
    with open('examples/pytorch/rdf2text/data/{}/raw/'.format(dataset_name)+part+'-all-notdelex-source.triple', 'r') as f:
        triples = [line.strip() for line in f]
    # outfileName = predfile.split('/')[-1].split('.')[0]+ '_relexicalised_predictions.txt'
    # with open(outfileName, 'w+', encoding='utf8') as f:
    relexoutputs = []
    for triple in triples:
        relexoutput = src_gens[triple]
        if lowercased:
            relexoutput = relexoutput.lower()
        relexoutputs.append(relexoutput)
        # f.write(relexoutput)

    return relexoutputs

class GCNModel(nn.Module):
    def __init__(self, vocab, config):
        super(GCNModel, self).__init__()
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

        self.loss_calc = Graph2SeqLoss(ignore_index=self.vocab.out_word_vocab.PAD,
                                       use_coverage=self.use_coverage,
                                       coverage_weight=config['coverage_loss_ratio'])

    def encode_init_node_feature(self, data):
        num_graph_nodes = []
        for g in data['graph_data']:
            num_graph_nodes.append(g.get_node_num())

        # graph embedding construction
        batch_gd = self.g2s.graph_topology(data['graph_data'])

        return batch_gd

    def forward(self, data, oov_dict=None, require_loss=True):
        # batch_gd = self.encode_init_node_feature(data)
        # prob, enc_attn_weights, coverage_vectors = self.g2s.encoder_decoder(batch_gd, data['graph_data'], oov_dict=oov_dict, tgt_seq=data['tgt_seq'])

        if require_loss:
            prob, enc_attn_weights, coverage_vectors = self.g2s(data['graph_data'],
                                                                oov_dict=oov_dict,
                                                                tgt_seq=data['tgt_seq'])

            tgt = data['tgt_seq']
            min_length = min(prob.shape[1], tgt.shape[1])
            prob = prob[:, :min_length, :]
            tgt = tgt[:, :min_length]
            loss = self.loss_calc(prob, label=tgt, enc_attn_weights=enc_attn_weights, coverage_vectors=coverage_vectors)
            return prob, loss * min_length / 2
        else:
            prob, enc_attn_weights, coverage_vectors = self.g2s(data['graph_data'],
                                                                oov_dict=oov_dict)
            return prob


class GCNGMPModel(nn.Module):
    def __init__(self, vocab, config):
        super(GCNGMPModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.use_coverage = self.config['decoder_args']['rnn_decoder_share']['use_coverage']

        # build Graph2Seq model
        self.g2s = GraphGMP2Seq.from_args(config, self.vocab, config['device'])

        if 'w2v' in self.g2s.graph_topology.embedding_layer.word_emb_layers:
            self.word_emb = self.g2s.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                            self.vocab.in_word_vocab.embeddings.shape[0],
                            self.vocab.in_word_vocab.embeddings.shape[1],
                            pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                            fix_emb=config['graph_construction_args']['node_embedding']['fix_word_emb'],
                            device=config['device']).word_emb_layer

        self.loss_calc = GraphGMP2SeqLoss(ignore_index=self.vocab.out_word_vocab.PAD,
                                          use_coverage=self.use_coverage,
                                          coverage_weight=config['coverage_loss_ratio'])

    def encode_init_node_feature(self, data):
        num_graph_nodes = []
        for g in data['graph_data']:
            num_graph_nodes.append(g.get_node_num())

        # graph embedding construction
        batch_gd = self.g2s.graph_topology(data['graph_data'])

        return batch_gd

    def forward(self, data, oov_dict=None, require_loss=True):
        # batch_gd = self.encode_init_node_feature(data)

        if require_loss:
            prob, enc_attn_weights, coverage_vectors = self.g2s.encoder_decoder(batch_gd,
                                                                                data['graph_data'],
                                                                                data['gmp_seq'],
                                                                                data['gmp_jump'],
                                                                                oov_dict=oov_dict,
                                                                                tgt_seq=data['tgt_seq'])

            tgt = data['tgt_seq']
            min_length = min(prob.shape[1], tgt.shape[1])
            prob = prob[:, :min_length, :]
            tgt = tgt[:, :min_length]
            loss = self.loss_calc(prob, label=tgt, enc_attn_weights=enc_attn_weights, coverage_vectors=coverage_vectors)
            return prob, loss * min_length / 2
        else:
            prob, enc_attn_weights, coverage_vectors = self.g2s.encoder_decoder(batch_gd,
                                                                                data['graph_data'],
                                                                                data['gmp_seq'],
                                                                                data['gmp_jump'],
                                                                                oov_dict=oov_dict)
            return prob


class GMPModel(nn.Module):
    def __init__(self, vocab, config):
        super(GMPModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.use_coverage = self.config['decoder_args']['rnn_decoder_share']['use_coverage']

        # build model
        self.g2s = GMP2Seq.from_args(config, self.vocab, config['device'])

        if 'w2v' in self.g2s.graph_topology.embedding_layer.word_emb_layers:
            self.word_emb = self.g2s.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                            self.vocab.in_word_vocab.embeddings.shape[0],
                            self.vocab.in_word_vocab.embeddings.shape[1],
                            pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                            fix_emb=config['graph_construction_args']['node_embedding']['fix_word_emb'],
                            device=config['device']).word_emb_layer

        self.loss_calc = Graph2SeqLoss(ignore_index=self.vocab.out_word_vocab.PAD,
                                       use_coverage=self.use_coverage,
                                       coverage_weight=config['coverage_loss_ratio'])

    def encode_init_node_feature(self, data):
        num_graph_nodes = []
        for g in data['graph_data']:
            num_graph_nodes.append(g.get_node_num())

        # graph embedding construction
        batch_gd = self.g2s.graph_topology(data['graph_data'])

        return batch_gd

    def forward(self, data, oov_dict=None, require_loss=True):
        batch_gd = self.encode_init_node_feature(data)

        if require_loss:
            prob, enc_attn_weights, coverage_vectors = self.g2s.encoder_decoder(batch_gd,
                                                                                data['graph_data'],
                                                                                data['gmp_seq'],
                                                                                data['gmp_jump'],
                                                                                oov_dict=oov_dict,
                                                                                tgt_seq=data['tgt_seq'])

            tgt = data['tgt_seq']
            min_length = min(prob.shape[1], tgt.shape[1])
            prob = prob[:, :min_length, :]
            tgt = tgt[:, :min_length]
            loss = self.loss_calc(prob, label=tgt, enc_attn_weights=enc_attn_weights, coverage_vectors=coverage_vectors)
            return prob, loss * min_length / 2
        else:
            prob, enc_attn_weights, coverage_vectors = self.g2s.encoder_decoder(batch_gd,
                                                                                data['graph_data'],
                                                                                data['gmp_seq'],
                                                                                data['gmp_jump'],
                                                                                oov_dict=oov_dict)
            return prob


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
                # dynamic_init_topology_builder
                raise RuntimeError('Define your own dynamic_init_topology_builder')
        else:
            raise NotImplementedError("Define your topology builder.")

        # dataset = WebNLGDataset(root_dir=self.config['graph_construction_args']['graph_construction_share']['root_dir'],
        #                         pretrained_word_emb_file=self.config['pre_word_emb_file'],
        #                         merge_strategy=self.config['graph_construction_args']['graph_construction_private']['merge_strategy'],
        #                         edge_strategy=self.config['graph_construction_args']["graph_construction_private"]['edge_strategy'],
        #                         max_word_vocab_size=self.config['top_word_vocab'],
        #                         min_word_vocab_freq=self.config['min_word_freq'],
        #                         word_emb_size=self.config['word_emb_size'],
        #                         share_vocab=self.config['share_vocab'],
        #                         seed=self.config['seed'],
        #                         graph_type=graph_type,
        #                         topology_builder=topology_builder,
        #                         topology_subdir=self.config['graph_construction_args']['graph_construction_share']['topology_subdir'],
        #                         dynamic_graph_type=self.config['graph_construction_args']['graph_construction_share']['graph_type'],
        #                         dynamic_init_topology_builder=dynamic_init_topology_builder,
        #                         dynamic_init_topology_aux_args={'dummy_param': 0},
        #                         thread_number=4,
        #                         port=9000,
        #                         timeout=15000,
        #                         tokenizer=None)

        dataset =  WebNLGGMPDataset(root_dir=self.config['graph_construction_args']['graph_construction_share']['root_dir'],
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
                                    thread_number=4,
                                    port=9000,
                                    timeout=15000,
                                    tokenizer=None)

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
        if self.config['model_type'] == 'gcn':
            self.model = GCNModel(self.vocab, self.config).to(self.config['device'])
        elif self.config['model_type'] == 'gmp':
            self.model = GMPModel(self.vocab, self.config).to(self.config['device'])
        elif self.config['model_type'] == 'gcn_gmp':
            self.model = GCNGMPModel(self.vocab, self.config).to(self.config['device'])
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
                    # if self.config['use_bert'] and self.config.get('finetune_bert', None):
                    #     parameters += [p for p in self.config['bert_model'].parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

                self.optimizer.step()
                train_loss.append(loss.item())
                print('Epoch = {}, Step = {}, Loss = {:.3f}'.format(epoch, i, loss.item()))

                pred = torch.max(logits, dim=-1)[1].cpu()
                dur.append(time.time() - t0)
                # if i==20:
                #     break

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

            # scores = self.evaluate_predictions(all_refs, hyps)
            # print('BLEU1 = {}, BLEU2 = {}, BLEU3 = {}, BLEU4 = {}, METEOR = {}'.format(scores['BLEU_1'],
            #                                                                            scores['BLEU_2'],
            #                                                                            scores['BLEU_3'],
            #                                                                            scores['BLEU_4'],
            #                                                                            scores['METEOR']))

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
