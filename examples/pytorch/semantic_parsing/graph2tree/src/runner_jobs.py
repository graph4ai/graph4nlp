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
from graph4nlp.pytorch.datasets.geo import GeoDatasetForTree

from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import NodeEmbeddingBasedRefinedGraphConstruction

from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE
from graph4nlp.pytorch.modules.graph_embedding.gcn import GCN

from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import \
    StdTreeDecoder
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import BeamSearchStrategy

from graph4nlp.pytorch.modules.utils.tree_utils import to_cuda

from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder, create_mask
from graph4nlp.pytorch.modules.utils.tree_utils import DataLoaderForGraphEncoder, Tree, Vocab, to_cuda

import warnings

warnings.filterwarnings('ignore')

class Graph2Tree(nn.Module):
    def __init__(self, src_vocab,
                 tgt_vocab,
                 use_copy,
                 enc_hidden_size,
                 dec_hidden_size,
                 dec_dropout_input,
                 dec_dropout_output,
                 dropout_for_word_embedding,
                 enc_dropout_for_feature,
                 enc_dropout_for_attn,
                 direction_option,
                 input_size,
                 output_size,
                 device,
                 teacher_force_ratio,
                 max_dec_seq_length,
                 max_dec_tree_depth,
                 graph_construction_type,
                 gnn_type,
                 rnn_type):
        super(Graph2Tree, self).__init__()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.use_copy = use_copy
        self.use_edge_weight = False

        # embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
        #                    'seq_info_encode_strategy': "bilstm"}

        embedding_style = {'single_token_item': True,
                           'emb_strategy': "w2v_bilstm",
                           'num_rnn_layers': 1
                           }

        if graph_construction_type == "DependencyGraph":
            self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                vocab=self.src_vocab,
                                                                hidden_size=enc_hidden_size, word_dropout=dropout_for_word_embedding, rnn_dropout=0.1, device=device,
                                                                fix_word_emb=False)
        elif graph_construction_type == "ConstituencyGraph":
            self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                                                vocab=self.src_vocab,
                                                                hidden_size=enc_hidden_size, word_dropout=dropout_for_word_embedding, rnn_dropout=0.1, device=device,
                                                                fix_word_emb=False)
        elif graph_construction_type == "DynamicGraph_node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                self.src_vocab,
                embedding_style,
                sim_metric_type='weighted_cosine',
                num_heads=1,
                top_k_neigh=None,
                epsilon_neigh=0.5,
                smoothness_ratio=0.1,
                connectivity_ratio=0.05,
                sparsity_ratio=0.1,
                input_size=enc_hidden_size,
                hidden_size=enc_hidden_size,
                fix_word_emb=False,
                word_dropout=dropout_for_word_embedding,
                rnn_dropout=0.1,
                device=device)
            self.use_edge_weight = True
        elif graph_construction_type == "DynamicGraph_node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                self.src_vocab,
                embedding_style,
                0.2,
                sim_metric_type="weighted_cosine",
                num_heads=1,
                top_k_neigh=None,
                epsilon_neigh=0.5,
                smoothness_ratio=0.1,
                connectivity_ratio=0.05,
                sparsity_ratio=0.1,
                input_size=enc_hidden_size,
                hidden_size=enc_hidden_size,
                fix_word_emb=False,
                word_dropout=dropout_for_word_embedding,
                rnn_dropout=0.1,
                device=device)
            self.use_edge_weight = True
        else:
            raise NotImplementedError()
            # self.graph_topology = NodeEmbeddingBasedGraphConstruction(word_vocab=self.src_vocab, 
            #                                                     embedding_styles=embedding_style, 
            #                                                     input_size=enc_hidden_size, 
            #                                                     hidden_size=enc_hidden_size,
            #                                                     top_k_neigh=200,
            #                                                     device=device)

        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer

        if gnn_type == "GAT":
            self.encoder = GAT(1, enc_hidden_size, enc_hidden_size, enc_hidden_size, [1],
                                direction_option=direction_option, feat_drop=enc_dropout_for_feature, 
                                attn_drop=enc_dropout_for_attn, activation=F.relu, residual=True)
        elif gnn_type == "GGNN":
            self.encoder = GGNN(1, enc_hidden_size, enc_hidden_size,
                                dropout=enc_dropout_for_feature, use_edge_weight=self.use_edge_weight, 
                                direction_option=direction_option)
        elif gnn_type == "SAGE":
            # aggregate type: 'mean','gcn','pool','lstm'
            self.encoder = GraphSAGE(1, enc_hidden_size, enc_hidden_size, enc_hidden_size,
                                    'lstm', direction_option=direction_option, feat_drop=enc_dropout_for_feature,
                                    activation=F.relu, bias=True, use_edge_weight=self.use_edge_weight)
        elif gnn_type == "GCN":
            self.encoder = GCN(1,
                                enc_hidden_size,
                                enc_hidden_size,
                                enc_hidden_size,
                                direction_option=direction_option,
                                norm="both",
                                activation=F.relu,
                                use_edge_weight=self.use_edge_weight)
        else:
            print("Wrong gnn type, please use GAT GGNN or SAGE")
            raise NotImplementedError()
        self.criterion = nn.NLLLoss(size_average=False)

        self.decoder = StdTreeDecoder(attn_type="uniform",
                                      embeddings=self.word_emb,
                                      enc_hidden_size=enc_hidden_size,
                                      dec_emb_size=self.tgt_vocab.embedding_dims,
                                      dec_hidden_size=dec_hidden_size,
                                      output_size=output_size,
                                      device=device,
                                      criterion=self.criterion,
                                      teacher_force_ratio=teacher_force_ratio,
                                      use_sibling=False,
                                      use_attention=True,
                                      use_copy=self.use_copy,
                                      use_coverage=True,
                                      fuse_strategy="average",
                                      num_layers=1,
                                      dropout_input=dec_dropout_input,
                                      dropout_output=dec_dropout_output,
                                      rnn_type=rnn_type,
                                      max_dec_seq_length=max_dec_seq_length,
                                      max_dec_tree_depth=max_dec_tree_depth,
                                      tgt_vocab=self.tgt_vocab)

    def forward(self, graph_list, tgt_tree_batch):
        batch_graph = self.graph_topology(graph_list)
        batch_graph.to(self.device)
        batch_graph = self.encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

        loss = self.decoder(from_batch(batch_graph), tgt_tree_batch=tgt_tree_batch, enc_batch=DataLoaderForGraphEncoder.get_input_text_batch(graph_list, self.use_copy, self.src_vocab))
        return loss

    def init(self, init_weight):
        to_cuda(self.encoder, self.device)
        to_cuda(self.decoder, self.device)

        print('--------------------------------------------------------------')
        for name, param in self.named_parameters():
            # print(name, param.size())
            if param.requires_grad:
                if ("word_embedding" in name) or ("word_emb_layer" in name) or ("bert_embedding" in name):
                    pass
                else:
                    if len(param.size()) >= 2:
                        if "rnn" in name:
                            init.orthogonal_(param)
                        else:
                            init.xavier_uniform_(param, gain=1.0)
                    else:
                        init.uniform_(param, -init_weight, init_weight)
        # print('--------------------------------------------------------------')


class Jobs:
    def __init__(self, opt=None):
        super(Jobs, self).__init__()
        self.opt = opt
        
        seed = opt.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if opt.gpuid == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(opt.gpuid))
        self.use_copy = opt.use_copy
        self.data_dir = opt.data_dir
        self.checkpoint_dir = opt.checkpoint_dir

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        use_copy = self.use_copy

        if self.opt.graph_construction_type == "DependencyGraph":
            dataset = JobsDatasetForTree(root_dir=self.data_dir,
                                topology_builder=DependencyBasedGraphConstruction,
                                topology_subdir='DependencyGraph', edge_strategy='as_node',
                                share_vocab=use_copy, enc_emb_size=self.opt.enc_emb_size,
                                dec_emb_size=self.opt.tgt_emb_size)

        elif self.opt.graph_construction_type == "ConstituencyGraph":
            dataset = JobsDatasetForTree(root_dir=self.data_dir,
                                topology_builder=ConstituencyBasedGraphConstruction,
                                topology_subdir='ConstituencyGraph', share_vocab=use_copy,
                                enc_emb_size=self.opt.enc_emb_size, dec_emb_size=self.opt.tgt_emb_size)

        elif self.opt.graph_construction_type == "DynamicGraph_node_emb":
            dataset = JobsDatasetForTree(root_dir=self.data_dir, seed=self.opt.seed, word_emb_size=self.opt.enc_emb_size,
                                topology_builder=NodeEmbeddingBasedGraphConstruction,
                                topology_subdir='DynamicGraph_node_emb', graph_type='dynamic',
                                dynamic_graph_type='node_emb', share_vocab=use_copy,
                                enc_emb_size=self.opt.enc_emb_size, dec_emb_size=self.opt.tgt_emb_size)

        elif self.opt.graph_construction_type == "DynamicGraph_node_emb_refined":
            if self.opt.dynamic_init_graph_type is None or self.opt.dynamic_init_graph_type == 'line':
                dynamic_init_topology_builder = None
            elif self.opt.dynamic_init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif self.opt.dynamic_init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            else:
                # dynamic_init_topology_builder
                raise RuntimeError('Define your own dynamic_init_topology_builder')
            dataset = JobsDatasetForTree(root_dir=self.data_dir, seed=self.opt.seed, word_emb_size=self.opt.enc_emb_size,
                                topology_builder=NodeEmbeddingBasedRefinedGraphConstruction,
                                topology_subdir='DynamicGraph_node_emb_refined', graph_type='dynamic',
                                dynamic_graph_type='node_emb_refined', share_vocab=use_copy,
                                enc_emb_size=self.opt.enc_emb_size, dec_emb_size=self.opt.tgt_emb_size,
                                dynamic_init_topology_builder=dynamic_init_topology_builder)
        else:
            raise NotImplementedError


        self.train_data_loader = DataLoaderForGraphEncoder(
            use_copy=use_copy, data=dataset.train, dataset=dataset, mode="train", batch_size=20, device=self.device)
        print("train sample size:", len(self.train_data_loader.data))
        self.test_data_loader = DataLoaderForGraphEncoder(
            use_copy=use_copy, data=dataset.test, dataset=dataset, mode="test", batch_size=1, device=self.device)
        print("test sample size:", len(self.test_data_loader.data))

        self.src_vocab = self.train_data_loader.src_vocab
        self.tgt_vocab = self.train_data_loader.tgt_vocab
        if use_copy:
            self.share_vocab = self.train_data_loader.share_vocab
        print("---Loading data done---\n")

    def _build_model(self):
        '''For encoder-decoder'''
        self.model = Graph2Tree(src_vocab=self.src_vocab,
                                tgt_vocab=self.tgt_vocab,
                                use_copy=self.use_copy,
                                enc_hidden_size=self.opt.enc_hidden_size,
                                dec_hidden_size=self.opt.dec_hidden_size,
                                dec_dropout_input=self.opt.dec_dropout_input,
                                dec_dropout_output=self.opt.dec_dropout_output,
                                dropout_for_word_embedding=self.opt.dropout_for_word_embedding,
                                enc_dropout_for_feature=self.opt.enc_dropout_for_feature,
                                enc_dropout_for_attn=self.opt.enc_dropout_for_attn,
                                direction_option=self.opt.direction_option,
                                input_size=self.src_vocab.vocab_size,
                                output_size=self.tgt_vocab.vocab_size,
                                device=self.device,
                                teacher_force_ratio=self.opt.teacher_force_ratio,
                                max_dec_seq_length=self.opt.max_dec_seq_length,
                                max_dec_tree_depth=self.opt.max_dec_tree_depth_for_train,
                                graph_construction_type=self.opt.graph_construction_type,
                                gnn_type=self.opt.gnn_type,
                                rnn_type=self.opt.rnn_type)
        self.model.init(self.opt.init_weight)
        self.model = to_cuda(self.model, self.device)
        print(self.model)

    def _build_optimizer(self):
        optim_state = {"learningRate": self.opt.learning_rate,
                       "weight_decay": self.opt.weight_decay}
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            parameters, lr=optim_state['learningRate'], weight_decay=optim_state['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.opt.max_epochs//3, gamma=0.5)

    def train_epoch(self, epoch):
        loss_to_print = 0
        for i in range(self.train_data_loader.num_batch):
            self.optimizer.zero_grad()
            batch_graph_list, _, batch_tree_list = self.train_data_loader.random_batch()
            loss = self.model(batch_graph_list, batch_tree_list)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)
            self.optimizer.step()
            loss_to_print += loss
        return loss_to_print/self.train_data_loader.num_batch


    def train(self):
        best_acc = -1

        print("-------------\nStarting training.")
        for epoch in range(self.opt.max_epochs):
            self.model.train()
            loss_to_print = self.train_epoch(epoch)
            # self.scheduler.step()
            print("epochs = {}, train_loss = {}".format(epoch, loss_to_print))
            # print(self.scheduler.get_lr())
            if epoch > 20:
                # torch.save(checkpoint, "{}/g2t".format(self.checkpoint_dir) + str(i))
                # pickle.dump(checkpoint, open("{}/g2t".format(self.checkpoint_dir) + str(i), "wb"))
                test_acc = self.eval((self.model))
                if test_acc > best_acc:
                    best_acc = test_acc
        print(best_acc)

    def eval(self, model):
        device = model.device

        max_dec_seq_length = 50
        max_dec_tree_depth = 20
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

            # get indexed tgt sequence
            reference = model.tgt_vocab.get_symbol_idx_for_list(x[1].split())

            # get input graph list
            input_graph_list = [x[0]]

            # get src sequence
            input_word_list = DataLoaderForGraphEncoder.get_input_text_batch(input_graph_list, use_copy, model.src_vocab)
            if input_word_list:
                input_word_list = to_cuda(input_word_list, device)

            candidate = do_generate(use_copy, enc_hidden_size, dec_hidden_size, model, input_graph_list, input_word_list,
                                    self.test_data_loader.src_vocab, self.test_data_loader.tgt_vocab, device, max_dec_seq_length, max_dec_tree_depth)
            candidate = [int(c) for c in candidate]
            num_left_paren = sum(
                1 for c in candidate if self.test_data_loader.tgt_vocab.idx2symbol[int(c)] == "(")
            num_right_paren = sum(
                1 for c in candidate if self.test_data_loader.tgt_vocab.idx2symbol[int(c)] == ")")
            diff = num_left_paren - num_right_paren
            if diff > 0:
                for i in range(diff):
                    candidate.append(
                        self.test_data_loader.tgt_vocab.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]
            ref_str = convert_to_string(
                reference, self.test_data_loader.tgt_vocab)
            cand_str = convert_to_string(
                candidate, self.test_data_loader.tgt_vocab)
            reference_list.append(reference)
            candidate_list.append(candidate)
            # print(cand_str)

        test_acc = compute_tree_accuracy(
            candidate_list, reference_list, self.test_data_loader.tgt_vocab)
        print("TEST ACCURACY = {}\n".format(test_acc))
        return test_acc


def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)

def do_generate(use_copy, enc_hidden_size, dec_hidden_size, model, input_graph_list, enc_w_list, word_manager, form_manager, device, max_dec_seq_length, max_dec_tree_depth, use_beam_search=True):
    # initialize the rnn state to all zeros
    prev_c = torch.zeros((1, dec_hidden_size), requires_grad=False)
    prev_h = torch.zeros((1, dec_hidden_size), requires_grad=False)
    if use_copy:
        enc_outputs = torch.zeros((1, enc_w_list.size(1), dec_hidden_size), requires_grad=False)

    batch_graph = model.graph_topology(input_graph_list)
    batch_graph.to(device)
    batch_graph = model.encoder(batch_graph)
    batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

    graph_node_embedding = model.decoder._extract_params(from_batch(batch_graph))['graph_node_embedding']
    graph_level_embedding = torch.max(graph_node_embedding, 1)[0]
    rnn_node_embedding = torch.zeros_like(graph_node_embedding, requires_grad=False)
    rnn_node_embedding = to_cuda(rnn_node_embedding, device)

    # assert(use_copy == False or graph_node_embedding.size() == enc_outputs.size())
    # assert(graph_level_embedding.size() == prev_c.size())

    enc_outputs = graph_node_embedding
    prev_c = graph_level_embedding
    prev_h = graph_level_embedding

    # print(form_manager.get_idx_symbol_for_list(enc_w_list[0]))

    # decode
    queue_decode = []
    queue_decode.append({"s": (prev_c, prev_h), "parent": 0, "child_index": 1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <= max_dec_tree_depth:
        s = queue_decode[head-1]["s"]
        parent_h = s[1]
        t = queue_decode[head-1]["t"]

        sibling_state = torch.zeros((1, dec_hidden_size), dtype=torch.float, requires_grad=False)
        sibling_state = to_cuda(sibling_state, device)

        flag_sibling = False
        for q_index in range(len(queue_decode)):
            if (head <= len(queue_decode)) and (q_index < head - 1) and (queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                flag_sibling = True
                sibling_index = q_index
        if flag_sibling:
            sibling_state = queue_decode[sibling_index]["s"][1]

        if head == 1:
            prev_word = torch.tensor([form_manager.get_symbol_idx(form_manager.start_token)], dtype=torch.long)
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('(')], dtype=torch.long)

        prev_word = to_cuda(prev_word, device)

        i_child = 1

        if use_copy:
            enc_context = None
            input_mask = create_mask(torch.LongTensor([enc_outputs.size(1)]*enc_outputs.size(0)), enc_outputs.size(1), device)
            decoder_state = (s[0].unsqueeze(0), s[1].unsqueeze(0))
        if not use_beam_search:
            while True:
                if not use_copy:
                    prediction, s, _ = model.decoder.decode_step(dec_single_input=prev_word,
                                                                            dec_single_state=s,
                                                                            memory=enc_outputs,
                                                                            parent_state=parent_h)

                    _, _prev_word = prediction.max(1)
                    prev_word = _prev_word
                else:
                    # print(form_manager.idx2symbol[np.array(prev_word)[0]])
                    decoder_embedded = model.decoder.embeddings(prev_word)
                    pred, decoder_state, _, _, enc_context = model.decoder.rnn(parent_h, sibling_state, decoder_embedded,
                                                                              decoder_state,
                                                                              enc_outputs.transpose(
                                                                                  0, 1),
                                                                              None, None, input_mask=input_mask,
                                                                              encoder_word_idx=enc_w_list,
                                                                              ext_vocab_size=model.decoder.embeddings.num_embeddings,
                                                                              log_prob=False,
                                                                              prev_enc_context=enc_context,
                                                                              encoder_outputs2=rnn_node_embedding.transpose(0, 1))

                    dec_next_state_1 = decoder_state[0].squeeze(0)
                    dec_next_state_2 = decoder_state[1].squeeze(0)

                    pred = torch.log(pred + 1e-31)
                    prev_word = pred.argmax(1)

                if int(prev_word[0]) == form_manager.get_symbol_idx(form_manager.end_token) or t.num_children >= max_dec_seq_length:
                    break
                elif int(prev_word[0]) == form_manager.get_symbol_idx(form_manager.non_terminal_token):
                    #print("we predicted N");exit()
                    if use_copy:
                        queue_decode.append({"s": (dec_next_state_1.clone(), dec_next_state_2.clone()), "parent": head, "child_index": i_child, "t": Tree()})
                    else:
                        queue_decode.append({"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index": i_child, "t": Tree()})
                    t.add_child(int(prev_word[0]))
                else:
                    t.add_child(int(prev_word[0]))
                i_child = i_child + 1
        else:
            beam_width = 4
            topk = 1        
            # decoding goes sentence by sentence
            assert(graph_node_embedding.size(0) == 1)
            beam_search_generator = BeamSearchStrategy(beam_size=beam_width, vocab=form_manager, decoder=model.decoder, rnn_type=None)
            for idx in range(graph_node_embedding.size(0)):
                decoded_results = beam_search_generator.beam_search_for_tree_decoding(decoder_initial_state=(s[0], s[1]), 
                                                                                        decoder_initial_input=prev_word,
                                                                                        parent_state=parent_h,
                                                                                        graph_node_embedding=enc_outputs,
                                                                                        rnn_node_embedding=rnn_node_embedding,
                                                                                        device=device,
                                                                                        topk=topk)
            generated_sentence = decoded_results[0][0]
            # print(" ".join(form_manager.get_idx_symbol_for_list([int(node_i.wordid.item()) for node_i in generated_sentence])))
            for node_i in generated_sentence:
                if int(node_i.wordid.item()) == form_manager.get_symbol_idx(form_manager.non_terminal_token):
                    queue_decode.append({"s": (node_i.h[0].clone(), node_i.h[1].clone()), "parent": head, "child_index": i_child, "t": Tree()})
                    t.add_child(int(node_i.wordid.item()))
                    i_child = i_child + 1
                elif int(node_i.wordid.item()) != form_manager.get_symbol_idx(form_manager.end_token) and \
                        int(node_i.wordid.item()) != form_manager.get_symbol_idx(form_manager.start_token) and \
                        int(node_i.wordid.item()) != form_manager.get_symbol_idx('('):
                    t.add_child(int(node_i.wordid.item()))
                    i_child = i_child + 1

        
        head = head + 1
    # refine the root tree (TODO, what is this doing?)
    for i in range(len(queue_decode)-1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"] -
                     1]["t"].children[cur["child_index"]-1] = cur["t"]
    return queue_decode[0]["t"].to_list(form_manager)

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
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        return all_same
    else:
        d1 = " ".join([form_manager.get_idx_symbol(x) for x in c1])
        d2 = " ".join([form_manager.get_idx_symbol(x) for x in c2])
        if get_split_comma(d1) == get_split_comma(d2):
            return True
        return False


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


if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    
    main_arg_parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-seed',type=int, default=123, help='torch manual random number generator seed')
    main_arg_parser.add_argument('-use_copy',type=bool, default=False, help='whether use copy mechanism')

    main_arg_parser.add_argument('-data_dir', type=str, 
            default='/home/lishucheng/Graph4AI/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/data/jobs', help='data path')
    main_arg_parser.add_argument('-checkpoint_dir',type=str, 
            default= '/home/lishucheng/Graph4AI/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/checkpoint_dir_jobs', help='output directory where checkpoints get written')
    
    main_arg_parser.add_argument('-gnn_type', type=str, default="SAGE")    
    main_arg_parser.add_argument('-rnn_type', type=str, default="lstm")    

    main_arg_parser.add_argument('-enc_emb_size', type=int, default=300)
    main_arg_parser.add_argument('-tgt_emb_size', type=int, default=300)

    main_arg_parser.add_argument('-enc_hidden_size', type=int, default=300)
    main_arg_parser.add_argument('-dec_hidden_size', type=int, default=300)

    main_arg_parser.add_argument('-graph_construction_type', type=str, default="ConstituencyGraph") # DynamicGraph_node_emb_refined, DynamicGraph_node_emb

    main_arg_parser.add_argument('-dynamic_init_graph_type', type=str, default="constituency") # "None, line, dependency, constituency"

    main_arg_parser.add_argument('-batch_size', type=int, default=20)

    main_arg_parser.add_argument('-dropout_for_word_embedding',type=float, default=0.1)

    main_arg_parser.add_argument('-enc_dropout_for_feature',type=float, default=0)
    main_arg_parser.add_argument('-enc_dropout_for_attn',type=float, default=0.1)
    
    main_arg_parser.add_argument('-direction_option',type=str, default="undirected")

    main_arg_parser.add_argument('-dec_dropout_input',type=float, default=0.1)
    main_arg_parser.add_argument('-dec_dropout_output',type=float, default=0.3)

    main_arg_parser.add_argument('-max_dec_seq_length',type=int, default=50)
    main_arg_parser.add_argument('-max_dec_tree_depth_for_train',type=int, default=50)
    main_arg_parser.add_argument('-max_dec_tree_depth_for_test',type=int, default=20)

    main_arg_parser.add_argument('-teacher_force_ratio',type=float, default=1.0)

    main_arg_parser.add_argument('-init_weight',type=float, default=0.08,help='initailization weight')
    main_arg_parser.add_argument('-learning_rate',type=float, default=1e-3,help='learning rate')
    main_arg_parser.add_argument('-weight_decay',type=float, default=0)

    main_arg_parser.add_argument('-max_epochs',type=int, default=150,help='number of full passes through the training data')
    main_arg_parser.add_argument('-grad_clip',type=int, default=5,help='clip gradients at this value')

    args = main_arg_parser.parse_args()

    runner = Jobs(opt=args)
    max_score = runner.train()

    end = time.time()
    print("total time: {} minutes\n".format((end - start)/60))