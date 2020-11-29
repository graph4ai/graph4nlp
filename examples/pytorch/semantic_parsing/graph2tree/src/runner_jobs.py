import os
import random
import time
import pickle
import argparse
import copy

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from stanfordcorenlp import StanfordCoreNLP

from graph4nlp.pytorch.data.data import GraphData, from_batch

from graph4nlp.pytorch.datasets.jobs import JobsDatasetForTree

from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import NodeEmbeddingBasedRefinedGraphConstruction

from graph4nlp.pytorch.modules.utils.tree_utils import to_cuda
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder, create_mask
from graph4nlp.pytorch.modules.utils.tree_utils import DataLoaderForGraphEncoder, Tree, Vocab, VocabForAll, to_cuda

import warnings

warnings.filterwarnings('ignore')

class Jobs:
    def __init__(self, opt=None):
        super(Jobs, self).__init__()
        self.opt = opt
        
        seed = opt["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if opt["gpuid"] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(opt["gpuid"]))
        self.use_copy = True if self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"] == 1 else False
        self.revectorization = True
        self.data_dir = opt["data_dir"]
        self.checkpoint_dir = opt["checkpoint_dir"]

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        use_copy = self.use_copy
        use_share_vocab = True

        if self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "dependency":
            topology_builder = DependencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "node_emb":
            topology_builder = NodeEmbeddingBasedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "node_emb_refined":
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_graph_type = self.opt["graph_construction_args"]["graph_construction_private"][
                "dynamic_init_graph_type"]
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

        dataset = JobsDatasetForTree(root_dir=self.data_dir,
                                         seed=self.opt["seed"],
                                         word_emb_size=self.opt["graph_embedding_args"]["graph_embedding_share"]["input_size"],
                                         edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"][
                                            "edge_strategy"],
                                         topology_builder=topology_builder,
                                         topology_subdir=self.opt["graph_construction_args"]["graph_construction_share"][
                                            "topology_subdir"], 
                                         graph_type=graph_type,
                                         dynamic_graph_type=self.opt["graph_construction_args"]["graph_construction_share"][
                                            "graph_type"],
                                         share_vocab=use_share_vocab,
                                         enc_emb_size=self.opt["graph_embedding_args"]["graph_embedding_share"]["input_size"],
                                         dec_emb_size=self.opt["decoder_args"]["rnn_decoder_share"]["input_size"],
                                         dynamic_init_topology_builder=dynamic_init_topology_builder, 
                                         device=self.device,
                                         min_freq=self.opt["min_freq"])

        self.train_data_loader = DataLoaderForGraphEncoder(
            use_copy=use_copy, use_share_vocab=use_share_vocab, data=dataset.train, dataset=dataset, mode="train", batch_size=self.opt["batch_size"], device=self.device)
        print("train sample size:", len(self.train_data_loader.data))
        self.test_data_loader = DataLoaderForGraphEncoder(
            use_copy=use_copy, use_share_vocab=use_share_vocab, data=dataset.test, dataset=dataset, mode="test", batch_size=1, device=self.device)
        print("test sample size:", len(self.test_data_loader.data))

        self.src_vocab = self.train_data_loader.src_vocab
        self.tgt_vocab = self.train_data_loader.tgt_vocab

        if use_share_vocab:
            self.share_vocab = self.train_data_loader.share_vocab
        print(self.share_vocab.symbol2idx)
        print("---Loading data done---\n")

    def _build_model(self):
        self.model = Graph2Tree.from_args(self.opt, VocabForAll(self.src_vocab, self.tgt_vocab, self.share_vocab), self.device)
        self.model.init(self.opt["init_weight"])
        self.model = to_cuda(self.model, self.device)
        print(self.model)

    def _build_optimizer(self):
        optim_state = {"learningRate": self.opt["learning_rate"],
                       "weight_decay": self.opt["weight_decay"]}
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            parameters, lr=optim_state['learningRate'], weight_decay=optim_state['weight_decay'])

    def prepare_ext_vocab(self, batch_graphs, src_vocab):
        oov_dict = copy.deepcopy(src_vocab)
        for g in batch_graphs:
            token_matrix = []
            for node_idx in range(g.get_node_num()):
                node_token = g.node_attributes[node_idx]['token']
                if (g.node_attributes[node_idx].get('type') == None or g.node_attributes[node_idx].get('type') == 0) \
                        and oov_dict.get_symbol_idx(node_token) == oov_dict.get_symbol_idx(oov_dict.unk_token):
                    oov_dict.add_symbol(node_token)
                token_matrix.append([oov_dict.get_symbol_idx(node_token)])
            token_matrix = torch.tensor(
                token_matrix, dtype=torch.long).to(self.device)
            g.node_features['token_id_oov'] = token_matrix
        return oov_dict

    def train_epoch(self, epoch):
        loss_to_print = 0
        for i in range(self.train_data_loader.num_batch):
            self.optimizer.zero_grad()
            batch_graph_list, _, batch_tree_list, batch_original_tree_list = self.train_data_loader.random_batch()

            oov_dict = self.prepare_ext_vocab(
                batch_graph_list, self.src_vocab) if self.use_copy else None


            if self.use_copy and self.revectorization:
                batch_tree_list_refined = []
                for item in batch_original_tree_list:
                    tgt_list = oov_dict.get_symbol_idx_for_list(item.strip().split())
                    tgt_tree = Tree.convert_to_tree(tgt_list, 0, len(tgt_list), oov_dict)
                    batch_tree_list_refined.append(tgt_tree)
            # for index in range(len(batch_tree_list_refined)):
            #     print("---------------------------------------")
            #     print(batch_tree_list[index])
            #     print(batch_tree_list_refined[index])
            # loss = self.model(batch_graph_list, batch_tree_list, oov_dict=oov_dict)
            loss = self.model(batch_graph_list, batch_tree_list_refined if self.use_copy else batch_tree_list, oov_dict=oov_dict)
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.opt["grad_clip"])
            self.optimizer.step()
            loss_to_print += loss
        return loss_to_print/self.train_data_loader.num_batch

    def train(self):
        best_acc = -1

        print("-------------\nStarting training.")
        for epoch in range(1, self.opt["max_epochs"]+1):
            self.model.train()
            loss_to_print = self.train_epoch(epoch)
            # self.scheduler.step()
            print("epochs = {}, train_loss = {:.3f}".format(epoch, loss_to_print))
            # print(self.scheduler.get_lr())
            if epoch > 20 and epoch % 10 == 0:
                # torch.save(checkpoint, "{}/g2t".format(self.checkpoint_dir) + str(i))
                # pickle.dump(checkpoint, open("{}/g2t".format(self.checkpoint_dir) + str(i), "wb"))
                test_acc = self.eval((self.model))
                if test_acc > best_acc:
                    best_acc = test_acc
        print("Best Acc: {:.3f}\n".format(best_acc))

    def eval(self, model):
        device = model.device

        max_dec_seq_length = self.opt["max_dec_seq_length"]
        max_dec_tree_depth = self.opt["max_dec_tree_depth_for_test"]
        
        use_copy = self.test_data_loader.use_copy
        enc_emb_size = model.src_vocab.embedding_dims
        tgt_emb_size = model.tgt_vocab.embedding_dims

        enc_hidden_size = model.decoder.enc_hidden_size
        dec_hidden_size = model.decoder.hidden_size

        model.eval()

        reference_list = []
        candidate_list = []

        data = self.test_data_loader.data

        for i in range(len(data)):
            x = data[i]

            # get input graph list
            input_graph_list = [x[0]]
            # if use_copy:
            oov_dict = self.prepare_ext_vocab(
                input_graph_list, self.test_data_loader.src_vocab)

            # get indexed tgt sequence
            if self.use_copy and self.revectorization:
                reference = oov_dict.get_symbol_idx_for_list(x[1].split())
                # reference = Tree.convert_to_tree(tmp_list, 0, len(tmp_list), oov_dict)
                eval_vocab = oov_dict

            else:
                reference = model.tgt_vocab.get_symbol_idx_for_list(x[1].split())
                eval_vocab = self.test_data_loader.tgt_vocab

            candidate = model.decoder.translate(use_copy,
                                                enc_hidden_size,
                                                dec_hidden_size,
                                                model,
                                                input_graph_list,
                                                self.test_data_loader.src_vocab,
                                                self.test_data_loader.tgt_vocab,
                                                device,
                                                max_dec_seq_length,
                                                max_dec_tree_depth,
                                                oov_dict=oov_dict)
            
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

            # for c in candidate:
            #     if c >= self.test_data_loader.tgt_vocab.vocab_size:
            #         print("====================")
            #         print(oov_dict.symbol2idx)
            #         print(cand_str)
            #         print(ref_str)
            #         print("====================")
            # print(cand_str)
            # print(ref_str)

            reference_list.append(reference)
            candidate_list.append(candidate)
            # print(cand_str)

        test_acc = compute_tree_accuracy(
            candidate_list, reference_list, eval_vocab)
        print("TEST ACCURACY = {:.3f}\n".format(test_acc))
        return test_acc


def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def get_split_comma(input_str):
    input_str = input_str.replace(",", " , ")
    input_list = [item.strip() for item in input_str.split()]
    ref_char = "$"
    for index in range(len(input_list)):
        if input_list[index] == ',':
            if input_list[:index].count('(') == input_list[:index].count(')'):
                if input_list[index+1:].count('(') == input_list[index+1:].count(')'):
                    if input_list[index] == ref_char:
                        raise RuntimeError
                    else:
                        input_list[index] = ref_char
    new_str = " ".join(input_list).split('$')
    result_set = set()
    for str_ in new_str:
        result_set.add(str_.strip())
    return result_set


def is_all_same(c1, c2, form_manager):
    all_same = True
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        if all_same:
            return True
    if len(c1) != len(c2) or all_same == False:
        d1 = " ".join([form_manager.get_idx_symbol(x) for x in c1])
        d2 = " ".join([form_manager.get_idx_symbol(x) for x in c2])
        if get_split_comma(d1) == get_split_comma(d2):
            # print(d1)
            # print(d2)
            # print("============")
            return True
        return False
    raise NotImplementedError("you should not arrive here!")

def compute_accuracy(candidate_list, reference_list, form_manager):
    if len(candidate_list) != len(reference_list):
        print("candidate list has length {}, reference list has length {}\n".format(
            len(candidate_list), len(reference_list)))

    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    for i in range(len_min):
        if is_all_same(candidate_list[i], reference_list[i], form_manager):
            c = c+1
        else:
            pass

    return c/float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    candidate_list = []
    for i in range(len(candidate_list_)):
        candidate_list.append(candidate_list_[i])
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(reference_list_[i])
    return compute_accuracy(candidate_list, reference_list, form_manager)


from .args_jobs import get_args

if __name__ == "__main__":
    start = time.time()
    runner = Jobs(opt=get_args())
    runner.train()
    end = time.time()
    print("total time: {} minutes\n".format((end - start)/60))