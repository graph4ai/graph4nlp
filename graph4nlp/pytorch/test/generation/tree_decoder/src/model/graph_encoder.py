import os
import time
import random
from . import nn_modules
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim


class GraphEncoder(nn.Module):
    def __init__(self, opt, input_size):
        super(GraphEncoder, self).__init__()
        self.opt = opt

        if opt.dropoutagg > 0:
            self.dropout = nn.Dropout(opt.dropoutagg)

        self.graph_encode_direction = opt.graph_encode_direction
        self.sample_size_per_layer = opt.sample_size_per_layer
        self.sample_layer_size = opt.sample_layer_size
        self.hidden_layer_dim = opt.rnn_size

        if self.opt.dropout_en_in > 0:
            self.input_dropout = nn.Dropout(opt.dropout_en_in)
        self.word_embedding_size = 300
        self.embedding = nn.Embedding(
            input_size, self.word_embedding_size, padding_idx=0)
        if opt.pretrain_flag == 1:
            self.embedding.weight.data = self.make_pretrained_embedding(
                self.embedding.weight.size(), opt)
            # self.embedding.weight.requires_grad = False

        self.fw_aggregators = []
        self.bw_aggregators = []

        self.fw_aggregator_0 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_1 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_2 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_3 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_4 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_5 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_6 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)

        self.bw_aggregator_0 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_1 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_2 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_3 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_4 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_5 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_6 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregators = [self.fw_aggregator_0, self.fw_aggregator_1, self.fw_aggregator_2,
                               self.fw_aggregator_3, self.fw_aggregator_4, self.fw_aggregator_5, self.fw_aggregator_6]
        self.bw_aggregators = [self.bw_aggregator_0, self.bw_aggregator_1, self.bw_aggregator_2,
                               self.bw_aggregator_3, self.bw_aggregator_4, self.bw_aggregator_5, self.bw_aggregator_6]

        self.Linear_hidden = nn.Linear(
            2 * self.hidden_layer_dim, self.hidden_layer_dim)

        self.concat = opt.concat

        self.using_gpu = False
        if self.opt.gpuid > -1:
            self.using_gpu = True

        self.embedding_bilstm = nn.LSTM(input_size=self.word_embedding_size, hidden_size=self.hidden_layer_dim/2,
                                        bidirectional=True, bias=True, batch_first=True, dropout=self.opt.dropout_en_out, num_layers=1)
        # self.embedding_bilstm = nn.LSTM(input_size=self.word_embedding_size, hidden_size=self.hidden_layer_dim, bidirectional=True, bias = True, batch_first = True, dropout= self.opt.dropout_en_out, num_layers=1)
        self.padding_vector = torch.randn(
            1, self.hidden_layer_dim, dtype=torch.float, requires_grad=True)

    def make_pretrained_embedding(self, embedding_size, opt):
        # use glove pretrained embedding and vocabulary to generate a embedding matrix
        data_dir = opt.data_dir
        min_freq = 2
        max_vocab_size = 15000
        torch.manual_seed(opt.seed)

        word2vec = pkl.load(
            open("{}/pretrain.pkl".format("../data/TextData"), "rb"))
        managers = pkl.load(open("{}/map.pkl".format(data_dir), "rb"))
        word_manager, form_manager = managers

        num_embeddings, embedding_dim = embedding_size
        weight_matrix = torch.zeros(
            (num_embeddings, embedding_dim), dtype=torch.float)
        cnt_change = 0
        for i in range(num_embeddings):
            word = word_manager.idx2symbol[i]
            if word in word2vec:
                weight_matrix[i] = torch.from_numpy(word2vec[word])
                cnt_change += 1
            else:
                weight_matrix[i] = torch.randn((embedding_dim, ))
        print(cnt_change)
        return weight_matrix

    def forward(self, graph_batch):
        fw_adj_info, bw_adj_info, feature_info, batch_nodes = graph_batch

        # print self.hidden_layer_dim

        if self.using_gpu > -1:
            fw_adj_info = fw_adj_info.cuda()
            bw_adj_info = bw_adj_info.cuda()
            feature_info = feature_info.cuda()
            batch_nodes = batch_nodes.cuda()

        feature_by_sentence = feature_info[:-1,
                                           :].view(batch_nodes.size()[0], -1)
        feature_sentence_vector = self.embedding(feature_by_sentence)
        if self.opt.dropout_en_in > 0:
            feature_sentence_vector = self.input_dropout(
                feature_sentence_vector)
        output_vector, (ht, _) = self.embedding_bilstm(feature_sentence_vector)

        # print output_vector.size()

        # output_vector = output_vector_[:, :, :self.hidden_layer_dim] + output_vector_[:, :, self.hidden_layer_dim:]
        # ht = output_vector_[-1, :, :self.hidden_layer_dim] + output_vector_[0, :, self.hidden_layer_dim:]

        # problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        # pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H

        feature_vector = output_vector.contiguous().view(-1, self.hidden_layer_dim)
        feature_embedded = torch.cat(
            [feature_vector, self.padding_vector.cuda()], 0)

        batch_size = feature_embedded.size()[0]
        node_repres = feature_embedded.view(batch_size, -1)

        fw_sampler = nn_modules.UniformNeighborSampler(fw_adj_info)
        bw_sampler = nn_modules.UniformNeighborSampler(bw_adj_info)
        nodes = batch_nodes.view(-1, )

        fw_hidden = F.embedding(nodes, node_repres)
        bw_hidden = F.embedding(nodes, node_repres)

        fw_sampled_neighbors = fw_sampler((nodes, self.sample_size_per_layer))
        bw_sampled_neighbors = bw_sampler((nodes, self.sample_size_per_layer))

        fw_sampled_neighbors_len = torch.tensor(0)
        bw_sampled_neighbors_len = torch.tensor(0)

        # begin sampling
        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 1
            if self.using_gpu and layer <= 6:
                self.fw_aggregators[layer] = self.fw_aggregators[layer].cuda()
            if layer == 0:
                neigh_vec_hidden = F.embedding(
                    fw_sampled_neighbors, node_repres)
                tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                tmp_mask = torch.sign(tmp_sum)
                fw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
            else:
                if self.using_gpu:
                    neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                        [1, dim_mul * self.hidden_layer_dim]).cuda()], 0))
                else:
                    neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                        [1, dim_mul * self.hidden_layer_dim])], 0))

            if layer > 6:
                fw_hidden = self.fw_aggregators[6](
                    (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            else:
                fw_hidden = self.fw_aggregators[layer](
                    (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))

            if self.graph_encode_direction == "bi":
                if self.using_gpu and layer <= 6:
                    self.bw_aggregators[layer] = self.bw_aggregators[layer].cuda(
                    )

                if layer == 0:
                    neigh_vec_hidden = F.embedding(
                        bw_sampled_neighbors, node_repres)
                    tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                    tmp_mask = torch.sign(tmp_sum)
                    bw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
                else:
                    if self.using_gpu:
                        neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                            [1, dim_mul * self.hidden_layer_dim]).cuda()], 0))
                    else:
                        neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                            [1, dim_mul * self.hidden_layer_dim])], 0))
                if self.opt.dropoutagg > 0:
                    bw_hidden = self.dropout(bw_hidden)
                    neigh_vec_hidden = self.dropout(neigh_vec_hidden)

                if layer > 6:
                    bw_hidden = self.bw_aggregators[6](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
                else:
                    bw_hidden = self.bw_aggregators[layer](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
        fw_hidden = fw_hidden.view(-1, batch_nodes.size()
                                   [1], self.hidden_layer_dim)

        if self.graph_encode_direction == "bi":
            bw_hidden = bw_hidden.view(-1, batch_nodes.size()
                                       [1], self.hidden_layer_dim)
            hidden = torch.cat([fw_hidden, bw_hidden], 2)
        else:
            hidden = fw_hidden

        pooled = torch.max(hidden, 1)[0]
        graph_embedding = pooled.view(-1, self.hidden_layer_dim)

        return hidden, graph_embedding, output_vector
