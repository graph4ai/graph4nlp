import os
import random
import time

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

from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE

from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import \
    StdTreeDecoder

from graph4nlp.pytorch.modules.utils.tree_utils import to_cuda

from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder, create_mask, dropout
from graph4nlp.pytorch.modules.utils.tree_utils import DataLoaderForGraphEncoder, Tree, Vocab, to_cuda

class Graph2Tree(nn.Module):
    def __init__(self, src_vocab,
                 tgt_vocab,
                 use_copy,
                 enc_hidden_size,
                 dec_hidden_size,
                 dec_dropout_input,
                 dec_dropout_output,
                 enc_dropout_input,
                 enc_dropout_output,
                 attn_dropout,
                 direction_option,
                 input_size,
                 output_size,
                 device,
                 teacher_force_ratio,
                 max_dec_seq_length,
                 max_dec_tree_depth):
        super(Graph2Tree, self).__init__()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.use_copy = use_copy

        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"}

        # TODO: specify two encoder RNN dropout ratios.
        
        self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=self.src_vocab,
                                                               hidden_size=enc_hidden_size, dropout=enc_dropout_input, use_cuda=(
                                                                   self.device != None),
                                                               fix_word_emb=False)
        # self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
        #                                                        vocab=self.src_vocab,
        #                                                        hidden_size=enc_hidden_size, dropout=enc_dropout_input, use_cuda=(
        #                                                            self.device != None),
        #                                                        fix_word_emb=False)
        # self.gnn = None

        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer


        # self.encoder = GAT(2, enc_hidden_size, enc_hidden_size, enc_hidden_size, [2, 1], direction_option=direction_option)
        self.encoder = GGNN(2, enc_hidden_size, enc_hidden_size, direction_option=direction_option)
        # self.encoder = GraphSAGE(2, enc_hidden_size, enc_hidden_size, enc_hidden_size, 'mean', direction_option=direction_option) # aggregate type: 'mean','gcn','pool','lstm'

        self.criterion = nn.NLLLoss(size_average=False)

        if not use_copy:
            attn_unit = AttnUnit(
                dec_hidden_size, output_size, "uniform", attn_dropout)
            self.decoder = StdTreeDecoder(attn=attn_unit,
                                          attn_type="uniform",
                                          embeddings=self.word_emb,
                                          enc_hidden_size=enc_hidden_size,
                                          dec_emb_size=self.tgt_vocab.embedding_dims,
                                          dec_hidden_size=dec_hidden_size,
                                          output_size=output_size,
                                          device=device,
                                          criterion=self.criterion,
                                          teacher_force_ratio=teacher_force_ratio,
                                          use_sibling=True,
                                          use_attention=True,
                                          use_copy=self.use_copy,
                                          use_coverage=True,
                                          fuse_strategy="average",
                                          num_layers=1,
                                          dropout_input=dec_dropout_input,
                                          dropout_output=dec_dropout_output,
                                          rnn_type="lstm",
                                          max_dec_seq_length=max_dec_seq_length,
                                          max_dec_tree_depth=max_dec_tree_depth,
                                          tgt_vocab=self.tgt_vocab)
        else:
            self.decoder = StdTreeDecoder(attn=None,
                                          attn_type="uniform",
                                          embeddings=self.word_emb,
                                          enc_hidden_size=enc_hidden_size,
                                          dec_emb_size=self.tgt_vocab.embedding_dims,
                                          dec_hidden_size=dec_hidden_size,
                                          output_size=output_size,
                                          device=device,
                                          criterion=self.criterion,
                                          teacher_force_ratio=teacher_force_ratio,
                                          use_sibling=True,
                                          use_attention=True,
                                          use_copy=self.use_copy,
                                          use_coverage=True,
                                          fuse_strategy="average",
                                          num_layers=1,
                                          dropout_input=dec_dropout_input,
                                          dropout_output=dec_dropout_output,
                                          rnn_type="lstm",
                                          max_dec_seq_length=max_dec_seq_length,
                                          max_dec_tree_depth=max_dec_tree_depth,
                                          tgt_vocab=self.tgt_vocab)

    def forward(self, graph_list, tgt_tree_batch):
        batch_graph = self.graph_topology(graph_list)
        batch_graph = self.encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

        loss = self.decoder(from_batch(batch_graph), tgt_tree_batch=tgt_tree_batch, enc_batch=DataLoaderForGraphEncoder.get_input_text_batch(graph_list, self.use_copy, self.src_vocab))
        return loss

    def init(self):
        to_cuda(self.encoder, self.device)
        to_cuda(self.decoder, self.device)

        init_weight = 0.08
        print('--------------------------------------------------------------')
        for name, param in self.named_parameters():
            print(name, param.size())
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
        print('--------------------------------------------------------------')


class Geo:
    def __init__(self, seed=1234, device=None, use_copy=True):
        super(Geo, self).__init__()
        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = device
        # self.use_copy = use_copy
        self.use_copy = False

        self.data_dir = "/Users/lishucheng/Desktop/g4nlp/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/data/geo"
        # self.data_dir = "/home/lishucheng/Graph4AI/graph4ai/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/data/geo"

        self.checkpoint_dir = "/Users/lishucheng/Desktop/g4nlp/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/checkpoint_dir_geo"
        # self.checkpoint_dir = "/home/lishucheng/Graph4AI/graph4ai/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/checkpoint_dir_geo"

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        use_copy = self.use_copy
        if use_copy:
            enc_emb_size = 300
            tgt_emb_size = 300
        else:
            enc_emb_size = 150
            tgt_emb_size = 150
        # dataset = GeoDatasetForTree(root_dir=self.data_dir,
        #                       topology_builder=ConstituencyBasedGraphConstruction,
        #                       topology_subdir='ConstituencyGraph', share_vocab=use_copy, enc_emb_size=enc_emb_size, dec_emb_size=tgt_emb_size)

        dataset = GeoDatasetForTree(root_dir=self.data_dir,
                              topology_builder=DependencyBasedGraphConstruction,
                              topology_subdir='DependencyGraph', share_vocab=use_copy, enc_emb_size=enc_emb_size, dec_emb_size=tgt_emb_size)

        self.train_data_loader = DataLoaderForGraphEncoder(
            use_copy=use_copy, dataset=dataset, mode="train", batch_size=20, device=self.device, ids_for_select=dataset.split_ids['train'])
        print("train sample size:", len(self.train_data_loader.data))
        self.test_data_loader = DataLoaderForGraphEncoder(
            use_copy=use_copy, dataset=dataset, mode="test", batch_size=1, device=self.device, ids_for_select=dataset.split_ids['test'])
        print("test sample size:", len(self.test_data_loader.data))

        self.src_vocab = self.train_data_loader.src_vocab
        self.tgt_vocab = self.train_data_loader.tgt_vocab
        if use_copy:
            self.share_vocab = self.train_data_loader.share_vocab
        print("---Loading data done---\n")

    def _build_model(self):
        '''For encoder-decoder'''
        # batch_size = self.train_data_loader.batch_size
        use_copy = self.use_copy
        if use_copy:
            input_size = self.share_vocab.vocab_size
            output_size = self.share_vocab.vocab_size
            # enc_hidden_size = 150
            enc_hidden_size = 300
            dec_hidden_size = 600
        else:
            input_size = self.src_vocab.vocab_size
            output_size = self.tgt_vocab.vocab_size
            enc_hidden_size = 150
            dec_hidden_size = 300

        enc_dropout_input = 0
        enc_dropout_output = 0

        dec_dropout_input = 0.1
        # dec_dropout_input = 0
        dec_dropout_output = 0.3

        attn_dropout = 0.1

        # teacher_force_ratio = 0.3
        teacher_force_ratio = 1

        max_dec_seq_length = 100
        max_dec_tree_depth = 100

        self.model = Graph2Tree(src_vocab=self.src_vocab,
                                tgt_vocab=self.tgt_vocab,
                                use_copy=self.use_copy,
                                enc_hidden_size=enc_hidden_size,
                                dec_hidden_size=dec_hidden_size,
                                dec_dropout_input=dec_dropout_input,
                                dec_dropout_output=dec_dropout_output,
                                enc_dropout_input=enc_dropout_input,
                                enc_dropout_output=enc_dropout_output,
                                attn_dropout=attn_dropout,
                                direction_option="bi_sep",
                                input_size=input_size,
                                output_size=output_size,
                                device=self.device,
                                teacher_force_ratio=teacher_force_ratio,
                                max_dec_seq_length=max_dec_seq_length,
                                max_dec_tree_depth=max_dec_tree_depth)
        self.model.init()

    def _build_optimizer(self):
        optim_state = {"learningRate": 1e-3,
                       "weight_decay":  1e-5}
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            parameters, lr=optim_state['learningRate'], weight_decay=optim_state['weight_decay'])

    def train(self, eva_every=1):
        '''eva_every: N, int. Do evaluation every N epochs.'''
        max_epochs = 300
        epoch = 0
        grad_clip = 5

        print("-------------\nStarting training.")
        self.model.train()

        iterations = max_epochs * self.train_data_loader.num_batch
        start_time = time.time()

        min_loss = 99999.
        best_index = -1
        test_index = -1

        checkpoint_dir = self.checkpoint_dir

        print("Batch number per Epoch:", self.train_data_loader.num_batch)
        print_every = eva_every * self.train_data_loader.num_batch

        loss_to_print = 0
        for i in range(iterations):
            epoch = i // self.train_data_loader.num_batch

            self.optimizer.zero_grad()
            batch_graph_list, _, batch_tree_list = self.train_data_loader.random_batch()

            loss = self.model(batch_graph_list, batch_tree_list)

            loss.backward()
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), grad_clip)
            self.optimizer.step()

            loss_to_print += loss

            if (i+1) % print_every == 0:
                end_time = time.time()
                print(("epochs = {}, {}/{}, train_loss = {}, time since last print = {}".format(epoch+1,  i+1,
                                                                                                iterations, (loss_to_print / print_every), (end_time - start_time)/60)))
                if loss_to_print < min_loss:
                    min_loss = loss_to_print
                    best_index = i
                # if i - best_index > (max_epochs//5)*self.train_data_loader.num_batch:
                # if True and epoch > 20:
                if True:
                    # print("Training loss does not decrease in {} epochs".format((max_epochs//5)))
                    checkpoint = {}
                    checkpoint["model"] = self.model
                    checkpoint["epoch"] = epoch
                    torch.save(checkpoint, "{}/g2t".format(self.checkpoint_dir) + str(i))
                    test_index = i
                    self.test(test_index)
                    # break

                loss_to_print = 0
                start_time = time.time()

            if loss != loss:
                print('loss is NaN.  This usually indicates a bug.')
                break
        self.test(test_index)

    def evaluate(self):
        device = None
        max_dec_seq_length = 220
        max_dec_tree_depth = 20
        use_copy = self.train_data_loader.use_copy

        if use_copy:
            enc_emb_size = 300
            tgt_emb_size = 300
            enc_hidden_size = 150
            dec_hidden_size = 300
        else:
            enc_emb_size = 150
            tgt_emb_size = 150
            enc_hidden_size = 150
            dec_hidden_size = 300

        encoder = self.encoder
        tree_decoder = self.decoder

        encoder.eval()
        tree_decoder.eval()

        reference_list = []
        candidate_list = []

        data = self.train_data_loader.data

        for i in range(len(data)):
            x = data[i]
            reference = torch.tensor(x[1], dtype=torch.long)
            input_word_list = x[0]
            candidate = do_generate(use_copy, enc_hidden_size, dec_hidden_size, encoder, tree_decoder, input_word_list,
                                    self.train_data_loader.src_vocab, self.train_data_loader.tgt_vocab, device, max_dec_seq_length, max_dec_tree_depth)
            candidate = [int(c) for c in candidate]
            num_left_paren = sum(
                1 for c in candidate if self.train_data_loader.tgt_vocab.idx2symbol[int(c)] == "(")
            num_right_paren = sum(
                1 for c in candidate if self.train_data_loader.tgt_vocab.idx2symbol[int(c)] == ")")
            diff = num_left_paren - num_right_paren
            if diff > 0:
                for i in range(diff):
                    candidate.append(
                        self.train_data_loader.tgt_vocab.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]
            ref_str = convert_to_string(
                reference, self.train_data_loader.tgt_vocab)
            cand_str = convert_to_string(
                candidate, self.train_data_loader.tgt_vocab)
            reference_list.append(reference)
            candidate_list.append(candidate)
            # print(cand_str)

        val_acc = compute_tree_accuracy(
            candidate_list, reference_list, self.test_data_loader.tgt_vocab)
        print("ACCURACY = {}\n".format(val_acc))
        return val_acc

    def test(self, fname_num):
        try:
            checkpoint = torch.load(
                "{}/g2t".format(self.checkpoint_dir) + str(fname_num))
        except BaseException:
            return FileNotFoundError()

        model = checkpoint["model"]

        device = model.device
        max_dec_seq_length = 100
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
            reference = torch.tensor(reference, dtype=torch.long)

            # get input graph list
            input_graph_list = [x[0]]

            # get src sequence
            input_word_list = DataLoaderForGraphEncoder.get_input_text_batch(input_graph_list, use_copy, model.src_vocab)

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


def do_generate(use_copy, enc_hidden_size, dec_hidden_size, model, input_graph_list, enc_w_list, word_manager, form_manager, device, max_dec_seq_length, max_dec_tree_depth):
    # initialize the rnn state to all zeros
    prev_c = torch.zeros((1, dec_hidden_size), requires_grad=False)
    prev_h = torch.zeros((1, dec_hidden_size), requires_grad=False)
    if use_copy:
        enc_outputs = torch.zeros((1, enc_w_list.size(1), dec_hidden_size), requires_grad=False)

    batch_graph = model.graph_topology(input_graph_list)
    batch_graph = model.encoder(batch_graph)
    batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

    graph_node_embedding = model.decoder._extract_params(from_batch(batch_graph))['graph_node_embedding']
    graph_level_embedding = torch.max(graph_node_embedding, 1)[0]
    rnn_node_embedding = torch.zeros_like(graph_node_embedding, requires_grad=False)

    assert(use_copy == False or graph_node_embedding.size() == enc_outputs.size())
    assert(graph_level_embedding.size() == prev_c.size())

    enc_outputs = graph_node_embedding
    prev_c = graph_level_embedding
    prev_h = graph_level_embedding

    # print(form_manager.get_idx_symbol_for_list(enc_w_list[0]))
    to_cuda(prev_c, device)
    to_cuda(prev_h, device)
    to_cuda(enc_outputs, device)

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

        to_cuda(prev_word, device)

        i_child = 1

        if use_copy:
            enc_context = None
            input_mask = create_mask(torch.LongTensor([enc_outputs.size(1)]*enc_outputs.size(0)), enc_outputs.size(1), device)
            decoder_state = (s[0].unsqueeze(0), s[1].unsqueeze(0))

        while True:
            if not use_copy:
                curr_c, curr_h = model.decoder.rnn(prev_word, s[0], s[1], parent_h, sibling_state)
                prediction = model.decoder.attention(enc_outputs, curr_h, torch.tensor(0))
                s = (curr_c, curr_h)
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
        head = head + 1
    # refine the root tree (TODO, what is this doing?)
    for i in range(len(queue_decode)-1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"] -
                     1]["t"].children[cur["child_index"]-1] = cur["t"]
    return queue_decode[0]["t"].to_list(form_manager)

class AttnUnit(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type, dropout):
        super(AttnUnit, self).__init__()
        self.hidden_size = hidden_size
        self.separate_attention = (attention_type != "uniform")

        if self.separate_attention == "separate_different_encoder_type":
            self.linear_att = nn.Linear(3*self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.linear_out = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0, 2, 1), attention)

        if self.separate_attention == "separate_different_encoder_type":
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0, 2, 1), attention_2)

        if self.separate_attention == "separate_different_encoder_type":
            hid = F.tanh(self.linear_att(torch.cat(
                (enc_attention.squeeze(2), enc_attention_2.squeeze(2), dec_s_top), 1)))
        else:
            hid = F.tanh(self.linear_att(
                torch.cat((enc_attention.squeeze(2), dec_s_top), 1)))
        h2y_in = hid

        h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)

        return pred


def is_all_same(c1, c2):
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        return all_same
    else:
        return False


def compute_accuracy(candidate_list, reference_list, form_manager):
    if len(candidate_list) != len(reference_list):
        print("candidate list has length {}, reference list has length {}\n".format(
            len(candidate_list), len(reference_list)))

    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    for i in range(len_min):
        if is_all_same(candidate_list[i], reference_list[i]):
            c = c+1
        else:
            pass

    return c/float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    candidate_list = []
    for i in range(len(candidate_list_)):
        candidate_list.append(Tree.norm_tree(
            candidate_list_[i], form_manager).to_list(form_manager))
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(Tree.norm_tree(
            reference_list_[i], form_manager).to_list(form_manager))
    return compute_accuracy(candidate_list, reference_list, form_manager)


if __name__ == "__main__":
    runner = Geo()
    max_score = runner.train()
