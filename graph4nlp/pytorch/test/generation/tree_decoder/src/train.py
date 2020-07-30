import argparse
import os
import pickle as pkl
import random
import re
import time
import warnings
from .config import get_args

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim

from .model.graph_encoder import GraphEncoder
from .utils import data_utils, graph_utils
from .utils.tree import Tree


def get_dec_batch(dec_tree_batch, opt, using_gpu, form_manager):
    queue_tree = {}
    for i in range(1, opt.batch_size+1):
        queue_tree[i] = []
        queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})

    cur_index, max_index = 1,1
    dec_batch = {}
    # max_index: the max number of sequence decoder in one batch
    while (cur_index <= max_index):
        max_w_len = -1
        batch_w_list = []
        for i in range(1, opt.batch_size+1):
            w_list = []
            if (cur_index <= len(queue_tree[i])):
                t = queue_tree[i][cur_index - 1]["tree"]

                for ic in range (t.num_children):
                    if isinstance(t.children[ic], Tree):
                        w_list.append(4)
                        queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index, "child_index": ic + 1})
                    else:
                        w_list.append(t.children[ic])
                if len(queue_tree[i]) > max_index:
                    max_index = len(queue_tree[i])
            if len(w_list) > max_w_len:
                max_w_len = len(w_list)
            batch_w_list.append(w_list)
        dec_batch[cur_index] = torch.zeros((opt.batch_size, max_w_len + 2), dtype=torch.long)
        for i in range(opt.batch_size):
            w_list = batch_w_list[i]
            if len(w_list) > 0:
                for j in range(len(w_list)):
                    dec_batch[cur_index][i][j+1] = w_list[j]
                # add <S>, <E>
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = 1
                else:
                    dec_batch[cur_index][i][0] = form_manager.get_symbol_idx('(')
                dec_batch[cur_index][i][len(w_list) + 1] = 2

        if using_gpu:
            dec_batch[cur_index] = dec_batch[cur_index].cuda()
        cur_index += 1

    return dec_batch, queue_tree, max_index


def eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, word_manager, form_manager):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    attention_decoder_optimizer.zero_grad()
    enc_batch, enc_len_batch, dec_tree_batch = train_loader.random_batch()

    enc_max_len = enc_len_batch

    enc_outputs = torch.zeros((opt.batch_size, enc_max_len, encoder.hidden_layer_dim), requires_grad=True)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()

    fw_adj_info = torch.tensor(enc_batch['g_fw_adj'])
    bw_adj_info = torch.tensor(enc_batch['g_bw_adj'])
    feature_info = torch.tensor(enc_batch['g_ids_features'])
    batch_nodes = torch.tensor(enc_batch['g_nodes'])
    # batch_wordlen = torch.tensor(enc_batch['word_len'])

    # node_embedding, graph_embedding, structural_info = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes, batch_wordlen))
    node_embedding, graph_embedding, structural_info = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes))

    enc_outputs = node_embedding

    graph_cell_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    graph_hidden_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    if using_gpu:
        graph_cell_state = graph_cell_state.cuda()
        graph_hidden_state = graph_hidden_state.cuda()
    
    graph_cell_state = graph_embedding
    graph_hidden_state = graph_embedding

    dec_s = {}
    for i in range(opt.dec_seq_length + 1):
        dec_s[i] = {}
        for j in range(opt.dec_seq_length + 1):
            dec_s[i][j] = {}

    # queue_tree = {}
    # for i in range(1, opt.batch_size+1):
    #     queue_tree[i] = []
    #     queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})


    loss = 0
    cur_index = 1

    dec_batch, queue_tree, max_index = get_dec_batch(dec_tree_batch, opt, using_gpu, form_manager)

    while (cur_index <= max_index):
        for j in range(1, 3):
            dec_s[cur_index][0][j] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
            if using_gpu:
                dec_s[cur_index][0][j] = dec_s[cur_index][0][j].cuda()

        sibling_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
        if using_gpu:
                sibling_state = sibling_state.cuda()

        if cur_index == 1:
            for i in range(opt.batch_size):
                dec_s[1][0][1][i, :] = graph_cell_state[i]
                dec_s[1][0][2][i, :] = graph_hidden_state[i]

        else:
            for i in range(1, opt.batch_size+1):
                if (cur_index <= len(queue_tree[i])):
                    par_index = queue_tree[i][cur_index - 1]["parent"]
                    child_index = queue_tree[i][cur_index - 1]["child_index"]
                    
                    dec_s[cur_index][0][1][i-1,:] = \
                        dec_s[par_index][child_index][1][i-1,:]
                    dec_s[cur_index][0][2][i-1,:] = dec_s[par_index][child_index][2][i-1,:]

                flag_sibling = False
                for q_index in range(len(queue_tree[i])):
                    if (cur_index <= len(queue_tree[i])) and (q_index < cur_index - 1) and (queue_tree[i][q_index]["parent"] == queue_tree[i][cur_index - 1]["parent"]) and (queue_tree[i][q_index]["child_index"] < queue_tree[i][cur_index - 1]["child_index"]):
                        flag_sibling = True
                        # sibling_index = queue_tree[i][q_index]["child_index"]
                        sibling_index = q_index
                if flag_sibling:
                    sibling_state[i - 1, :] = dec_s[sibling_index][dec_batch[sibling_index].size(1) - 1][2][i - 1,:]
                
        parent_h = dec_s[cur_index][0][2]
        for i in range(dec_batch[cur_index].size(1) - 1):
            # dec_s[cur_index][i+1][1], dec_s[cur_index][i+1][2] = decoder(dec_batch[cur_index][:,i], dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h, sibling_state)
            dec_s[cur_index][i+1][1], dec_s[cur_index][i+1][2] = decoder(dec_batch[cur_index][:,i], dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h, opt.para_for_sibling*sibling_state)
            pred = attention_decoder(enc_outputs, dec_s[cur_index][i+1][2], structural_info)

            loss += criterion(pred, dec_batch[cur_index][:,i+1])
        cur_index = cur_index + 1

    # for q_i in range(len(queue_tree.keys())):
    #     if len(queue_tree[q_i + 1]) > 0:
    #         for q_j in queue_tree[q_i + 1]:
    #             print q_j
    #     print "----------------------"


    loss = loss / opt.batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(decoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(attention_decoder.parameters(),opt.grad_clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    attention_decoder_optimizer.step()
    return loss

def do_generate(encoder, decoder, attention_decoder, graph_input, word_manager, form_manager, opt, using_gpu, checkpoint):    
    prev_c = torch.zeros((1, encoder.hidden_layer_dim), requires_grad=False)
    prev_h = torch.zeros((1, encoder.hidden_layer_dim), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()

    graph_size = len(graph_input['g_nodes'][0])
    enc_outputs = torch.zeros((1, graph_size, encoder.hidden_layer_dim), requires_grad=False)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()

    if graph_input['g_fw_adj'] == []:
        return "None"
    fw_adj_info = torch.tensor(graph_input['g_fw_adj'])
    bw_adj_info = torch.tensor(graph_input['g_bw_adj'])
    feature_info = torch.tensor(graph_input['g_ids_features'])
    batch_nodes = torch.tensor(graph_input['g_nodes'])

    node_embedding, graph_embedding, structural_info = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes))
    enc_outputs = node_embedding
    prev_c = graph_embedding
    prev_h = graph_embedding

    queue_decode = []
    queue_decode.append({"s": (prev_c, prev_h), "parent":0, "child_index":1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <=100:
        s = queue_decode[head-1]["s"]
        parent_h = s[1]
        t = queue_decode[head-1]["t"]

        sibling_state = torch.zeros((1, encoder.hidden_layer_dim), dtype=torch.float, requires_grad=True)

        if using_gpu:
            sibling_state = sibling_state.cuda()
        flag_sibling = False
        for q_index in range(len(queue_decode)):
            if (head <= len(queue_decode)) and (q_index < head - 1) and (queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                flag_sibling = True
                # print "ok"
                # sibling_index = queue_tree[i][q_index]["child_index"]
                sibling_index = q_index
        if flag_sibling:
            sibling_state = queue_decode[sibling_index]["s"][1]

        # print parent_h.size()
        # print sibling_state.size()

        if head == 1:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long)
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('(')], dtype=torch.long)
        if using_gpu:
            prev_word = prev_word.cuda()
        i_child = 1
        while True:
            # print sibling_state.size()
            curr_c, curr_h = decoder(prev_word, s[0], s[1], parent_h, opt.para_for_sibling*sibling_state)
            prediction = attention_decoder(enc_outputs, curr_h, structural_info)
 
            s = (curr_c, curr_h)
            _, _prev_word = prediction.max(1)
            prev_word = _prev_word

            if int(prev_word[0]) == form_manager.get_symbol_idx('<E>') or t.num_children >= checkpoint["opt"].dec_seq_length:
                break
            elif int(prev_word[0]) == form_manager.get_symbol_idx('<N>'):
                queue_decode.append({"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index":i_child, "t": Tree()})
                t.add_child(int(prev_word[0]))
            else:
                t.add_child(int(prev_word[0]))
            i_child = i_child + 1
        head = head + 1
    for i in range(len(queue_decode)-1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"]-1]["t"].children[cur["child_index"]-1] = cur["t"]
    return queue_decode[0]["t"].to_list(form_manager)
