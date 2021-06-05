import os
import random
import time
import pickle
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader

from graph4nlp.pytorch.data.data import to_batch
from graph4nlp.pytorch.datasets.geo import GeoDatasetForTree
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.graph_embedding import *
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.utils.tree_utils import Tree, VocabForAll

import warnings
warnings.filterwarnings('ignore')


class Geo:
    def __init__(self, opt=None):
        super(Geo, self).__init__()
        self.opt = opt

        seed = self.opt["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.opt["gpuid"] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(self.opt["gpuid"]))

        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_share_vocab = self.opt["graph_construction_args"]["graph_construction_share"]["share_vocab"]
        self.data_dir = self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"]

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        graph_type = self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"]
        enc_emb_size = self.opt["graph_construction_args"]["node_embedding"]["input_size"]
        tgt_emb_size = self.opt["decoder_args"]["rnn_decoder_share"]["input_size"]
        topology_subdir = self.opt["graph_construction_args"]["graph_construction_share"]["topology_subdir"]
        if graph_type == "dependency":
            dataset = GeoDatasetForTree(root_dir=self.data_dir,
                                         topology_builder=DependencyBasedGraphConstruction,
                                         topology_subdir=topology_subdir, 
                                         edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"]["edge_strategy"],
                                         graph_type='static',
                                         share_vocab=self.use_share_vocab, 
                                         enc_emb_size=enc_emb_size,
                                         dec_emb_size=tgt_emb_size,
                                         min_word_vocab_freq=self.opt["min_freq"],
                                         pretrained_word_emb_name=self.opt["pretrained_word_emb_name"],
                                         pretrained_word_emb_url=self.opt["pretrained_word_emb_url"], 
                                         pretrained_word_emb_cache_dir=self.opt["pretrained_word_emb_cache_dir"])

        elif graph_type == "constituency":
            dataset = GeoDatasetForTree(root_dir=self.data_dir,
                                         topology_builder=ConstituencyBasedGraphConstruction,
                                         topology_subdir=topology_subdir, 
                                         edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"]["edge_strategy"],
                                         graph_type='static',
                                         share_vocab=self.use_share_vocab, 
                                         enc_emb_size=enc_emb_size,
                                         dec_emb_size=tgt_emb_size,
                                         min_word_vocab_freq=self.opt["min_freq"],
                                         pretrained_word_emb_name=self.opt["pretrained_word_emb_name"],
                                         pretrained_word_emb_url=self.opt["pretrained_word_emb_url"], 
                                         pretrained_word_emb_cache_dir=self.opt["pretrained_word_emb_cache_dir"])
                                         
        elif graph_type == "node_emb":
            dataset = GeoDatasetForTree(root_dir=self.data_dir, 
                                         word_emb_size=enc_emb_size,
                                         topology_builder=NodeEmbeddingBasedGraphConstruction,
                                         topology_subdir=topology_subdir, 
                                         graph_type='dynamic',
                                         dynamic_graph_type=graph_type, 
                                         edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"]["edge_strategy"],
                                         share_vocab=self.use_share_vocab, 
                                         enc_emb_size=enc_emb_size,
                                         dec_emb_size=tgt_emb_size,
                                         min_word_vocab_freq=self.opt["min_freq"],
                                         pretrained_word_emb_name=self.opt["pretrained_word_emb_name"],
                                         pretrained_word_emb_url=self.opt["pretrained_word_emb_url"], 
                                         pretrained_word_emb_cache_dir=self.opt["pretrained_word_emb_cache_dir"])
    
        elif graph_type == "node_emb_refined":
            dynamic_init_graph_type = self.opt["graph_construction_args"]["graph_construction_private"]["dynamic_init_graph_type"]
            if dynamic_init_graph_type is None or dynamic_init_graph_type == 'line':
                dynamic_init_topology_builder = None
            elif dynamic_init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif dynamic_init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            else:
                # dynamic_init_topology_builder
                raise RuntimeError('Define your own dynamic_init_topology_builder')
            dataset = GeoDatasetForTree(root_dir=self.data_dir,
                                         word_emb_size=enc_emb_size,
                                         topology_builder=NodeEmbeddingBasedRefinedGraphConstruction,
                                         topology_subdir=topology_subdir,
                                         graph_type='dynamic',
                                         dynamic_graph_type=graph_type,
                                         share_vocab=self.use_share_vocab,
                                         enc_emb_size=enc_emb_size, 
                                         dec_emb_size=tgt_emb_size,
                                         dynamic_init_topology_builder=dynamic_init_topology_builder,
                                         min_word_vocab_freq=self.opt["min_freq"],
                                         pretrained_word_emb_name=self.opt["pretrained_word_emb_name"],
                                         pretrained_word_emb_url=self.opt["pretrained_word_emb_url"], 
                                         pretrained_word_emb_cache_dir=self.opt["pretrained_word_emb_cache_dir"])                                         
        else:
            raise NotImplementedError

        self.train_data_loader = DataLoader(dataset.train, batch_size=self.opt["batch_size"], shuffle=True, num_workers=1,
                                           collate_fn=dataset.collate_fn)
        self.test_data_loader = DataLoader(dataset.test, batch_size=1, shuffle=False, num_workers=1,
                                          collate_fn=dataset.collate_fn)
        self.src_vocab = dataset.src_vocab_model
        self.tgt_vocab = dataset.tgt_vocab_model
        if self.use_share_vocab:
            self.share_vocab = dataset.share_vocab_model
        self.vocab_model = VocabForAll(in_word_vocab=self.src_vocab, out_word_vocab=self.tgt_vocab, share_vocab=self.share_vocab)

    def _build_model(self):
        '''For encoder-decoder'''
        self.model = Graph2Tree.from_args(self.opt, vocab_model=self.vocab_model)
        self.model.init(self.opt["init_weight"])
        self.model.to(self.device)

    def _build_optimizer(self):
        optim_state = {"learningRate": self.opt["learning_rate"], "weight_decay": self.opt["weight_decay"]}
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=optim_state['learningRate'], weight_decay=optim_state['weight_decay'])

    def prepare_ext_vocab(self, batch_graph, src_vocab):
        oov_dict = copy.deepcopy(src_vocab)
        token_matrix = []
        for n in batch_graph.node_attributes:
            node_token = n['token']
            if (n.get('type') == None or n.get('type') == 0) and oov_dict.get_symbol_idx(node_token) == oov_dict.get_symbol_idx(oov_dict.unk_token):
                oov_dict.add_symbol(node_token)
            token_matrix.append(oov_dict.get_symbol_idx(node_token))
        batch_graph.node_features['token_id_oov'] = torch.tensor(token_matrix, dtype=torch.long).to(self.device)
        return oov_dict

    def train_epoch(self, epoch):
        loss_to_print = 0
        num_batch = len(self.train_data_loader)
        for step, data in enumerate(self.train_data_loader):
            batch_graph, batch_tree_list, batch_original_tree_list = data['graph_data'], data['dec_tree_batch'], data['original_dec_tree_batch']
            batch_graph = batch_graph.to(self.device)
            self.optimizer.zero_grad()
            oov_dict = self.prepare_ext_vocab(
                batch_graph, self.src_vocab) if self.use_copy else None

            if self.use_copy:
                batch_tree_list_refined = []
                for item in batch_original_tree_list:
                    tgt_list = oov_dict.get_symbol_idx_for_list(item.strip().split())
                    tgt_tree = Tree.convert_to_tree(tgt_list, 0, len(tgt_list), oov_dict)
                    batch_tree_list_refined.append(tgt_tree)
            loss = self.model(batch_graph, batch_tree_list_refined if self.use_copy else batch_tree_list, oov_dict=oov_dict)
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.opt["grad_clip"])
            self.optimizer.step()
            loss_to_print += loss
        return loss_to_print/num_batch

    def train(self):
        best_acc = -1

        print("-------------\nStarting training.")
        for epoch in range(1, self.opt["max_epochs"]+1):
            self.model.train()
            loss_to_print = self.train_epoch(epoch)
            print("epochs = {}, train_loss = {:.3f}".format(epoch, loss_to_print))
            if epoch > 2 and epoch % 5 == 0:
                test_acc = self.eval((self.model))
                if test_acc > best_acc:
                    best_acc = test_acc
        print("Best Acc: {:.3f}\n".format(best_acc))
        return best_acc

    def eval(self, model):
        from evaluation import convert_to_string, compute_tree_accuracy
        model.eval()
        reference_list = []
        candidate_list = []
        for data in self.test_data_loader:
            eval_input_graph, batch_tree_list, batch_original_tree_list = data['graph_data'], data['dec_tree_batch'], data['original_dec_tree_batch']
            eval_input_graph = eval_input_graph.to(self.device)
            oov_dict = self.prepare_ext_vocab(eval_input_graph, self.src_vocab)

            if self.use_copy:
                assert len(batch_original_tree_list) == 1
                reference = oov_dict.get_symbol_idx_for_list(batch_original_tree_list[0].split())
                eval_vocab = oov_dict
            else:
                assert len(batch_original_tree_list) == 1
                reference = model.tgt_vocab.get_symbol_idx_for_list(batch_original_tree_list[0].split())
                eval_vocab = self.tgt_vocab

            candidate = model.decoder.translate(model.use_copy,
                                                model.decoder.enc_hidden_size,
                                                model.decoder.hidden_size,
                                                model,
                                                eval_input_graph,
                                                self.src_vocab,
                                                self.tgt_vocab,
                                                self.device,
                                                self.opt["decoder_args"]["rnn_decoder_private"]["max_decoder_step"],
                                                self.opt["decoder_args"]["rnn_decoder_private"]["max_tree_depth"],
                                                oov_dict=oov_dict,
                                                use_beam_search=True,
                                                beam_size=self.opt["beam_size"])
            
            candidate = [int(c) for c in candidate]
            num_left_paren = sum(
                1 for c in candidate if eval_vocab.idx2symbol[int(c)] == "(")
            num_right_paren = sum(
                1 for c in candidate if eval_vocab.idx2symbol[int(c)] == ")")
            diff = num_left_paren - num_right_paren
            if diff > 0:
                for i in range(diff):
                    candidate.append(
                        self.test_data_loader.tgt_vocab.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]
            ref_str = convert_to_string(
                reference, eval_vocab)
            cand_str = convert_to_string(
                candidate, eval_vocab)

            reference_list.append(reference)
            candidate_list.append(candidate)
        test_acc = compute_tree_accuracy(
            candidate_list, reference_list, eval_vocab)
        print("TEST ACCURACY = {:.3f}\n".format(test_acc))
        return test_acc

if __name__ == "__main__":
    from config import get_args
    start = time.time()
    runner = Geo(opt=get_args())
    best_acc = runner.train()

    end = time.time()
    print("total time: {} minutes\n".format((end - start)/60))