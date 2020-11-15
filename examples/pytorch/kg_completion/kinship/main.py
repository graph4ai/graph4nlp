import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "5"

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.datasets.kinship import KinshipDataset
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT, GATLayer
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGELayer
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN, GGNNLayer
from graph4nlp.pytorch.modules.graph_embedding.gcn import GCNLayer
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
from graph4nlp.pytorch.modules.prediction.classification.kg_completion.DistMult import DistMult
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import dgl
import numpy as np
import copy
import random

from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import EmbeddingConstruction
from graph4nlp.pytorch.modules.loss.kg_loss import *


class RankingAndHits(EvaluationMetricBase):
    def __init__(self, model_path='best_graph2distmult', batch_size=64):
        super(RankingAndHits, self).__init__()
        self.batch_size = batch_size
        self.best_mrr = 0.
        self.best_hits1 = 0.
        self.best_hits10 = 0.
        self.model_path = model_path

    def calculate_scores(self, model, dataloader, name, kg_graph, device, save=True):
        print('')
        print('-' * 50)
        print(name)
        hits_left = []
        hits_right = []
        hits = []
        ranks = []
        ranks_left = []
        ranks_right = []
        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])
        with open('output_model.txt', 'w') as file:
            for data in dataloader:
                e1, rel, e2_multi, e2_multi_tensor_idx, \
                e2, rel_eval, e1_multi, e1_multi_tensor_idx = data

                e1 = e1.view(-1, 1).to(device)
                rel = rel.view(-1, 1).to(device)
                e2_multi = e2_multi.to(device)
                e2_multi_tensor_idx = e2_multi_tensor_idx.to(device)

                e2 = e2.view(-1, 1).to(device)
                rel_eval = rel_eval.view(-1, 1).to(device)
                e1_multi = e1_multi.to(device)
                e1_multi_tensor_idx = e1_multi_tensor_idx.to(device)
                pred1 = model(kg_graph, e1, rel, e2_multi, require_loss=False)
                pred2 = model(kg_graph, e2, rel_eval, e1_multi, require_loss=False)
                pred1, pred2 = pred1.data, pred2.data
                e1, e2 = e1.data, e2.data
                e2_multi1, e2_multi2 = e2_multi_tensor_idx.data, e1_multi_tensor_idx.data

                if e1.size()[0] != self.batch_size:
                    continue

                for i in range(self.batch_size):
                    # these filters contain ALL labels
                    filter1 = e2_multi1[i].long()
                    filter2 = e2_multi2[i].long()

                    num = e1[i, 0].item()
                    # save the prediction that is relevant
                    target_value1 = pred1[i, e2.cpu().numpy()[i, 0].item()].item()
                    target_value2 = pred2[i, e1.cpu().numpy()[i, 0].item()].item()
                    # zero all known cases (this are not interesting)
                    # this corresponds to the filtered setting
                    pred1[i][filter1] = 0.0
                    pred2[i][filter2] = 0.0
                    # write base the saved values
                    pred1[i][e2[i]] = target_value1
                    pred2[i][e1[i]] = target_value2

                    # print(e1[i, 0])

                # sort and rank
                max_values1, argsort1 = torch.sort(pred1, 1, descending=True)
                max_values2, argsort2 = torch.sort(pred2, 1, descending=True)

                argsort1 = argsort1.cpu().numpy()
                argsort2 = argsort2.cpu().numpy()
                for i in range(self.batch_size):
                    # find the rank of the target entities
                    rank1 = np.where(argsort1[i] == e2.cpu().numpy()[i, 0])[0][0]
                    if model.loss_name in ['SoftplusLoss', 'SigmoidLoss'] and max_values1[i][rank1] == max_values1[i][
                        0]:
                        rank1 = 0
                    rank2 = np.where(argsort2[i] == e1.cpu().numpy()[i, 0])[0][0]
                    if model.loss_name in ['SoftplusLoss', 'SigmoidLoss'] and max_values2[i][rank2] == max_values2[i][
                        0]:
                        rank2 = 0
                    # rank+1, since the lowest rank is rank 1 not rank 0
                    ranks.append(rank1 + 1)
                    ranks_left.append(rank1 + 1)
                    ranks.append(rank2 + 1)
                    ranks_right.append(rank2 + 1)

                    file.write(str(e2.cpu().numpy()[i, 0].item()) + '\t')
                    file.write(str(rank1 + 1) + '\n')
                    file.write(str(e1.cpu().numpy()[i, 0].item()) + '\t')
                    file.write(str(rank2 + 1) + '\n')

                    # this could be done more elegantly, but here you go
                    for hits_level in range(10):
                        if rank1 <= hits_level:
                            hits[hits_level].append(1.0)
                            hits_left[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)
                            hits_left[hits_level].append(0.0)

                        if rank2 <= hits_level:
                            hits[hits_level].append(1.0)
                            hits_right[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)
                            hits_right[hits_level].append(0.0)

        for i in range(10):
            print('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
            print('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
            print('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        print('Mean rank left: ', np.mean(ranks_left))
        print('Mean rank right: ', np.mean(ranks_right))
        print('Mean rank: ', np.mean(ranks))
        print('Mean reciprocal rank left: ', np.mean(1. / np.array(ranks_left)))
        print('Mean reciprocal rank right: ', np.mean(1. / np.array(ranks_right)))
        print('Mean reciprocal rank: ', np.mean(1. / np.array(ranks)))

        if np.mean(hits[0]) > self.best_hits1:
            self.best_hits1 = np.mean(hits[0])

        if np.mean(hits[9]) > self.best_hits10:
            self.best_hits10 = np.mean(hits[9])

        if np.mean(1. / np.array(ranks)) > self.best_mrr:
            self.best_mrr = np.mean(1. / np.array(ranks))
            if save:
                print('saving best model...')
                torch.save(model.state_dict(), self.model_path)

        print('-' * 50)
        print('')


# TODO: 1. initialize graph.node_features['edge_feat']/self.distmult.rel_emb with self.embedding_layer
# TODO: 2. learn graph.node_features['edge_emb'] from GNN (edge2node)

class Graph2DistMult(nn.Module):
    def __init__(self, args, vocab, device, num_entities, hidden_size=300, num_relations=None, direction_option='uni',
                 loss_name='BCELoss'):
        super(Graph2DistMult, self).__init__()
        self.args = args
        self.device = device
        self.vocab = vocab
        self.num_entities = num_entities

        self.node_emb = nn.Embedding(num_entities, hidden_size)

        embedding_style = {'single_token_item': False,
                           'emb_strategy': 'w2v_bilstm',
                           'num_rnn_layers': 1}

        self.embedding_layer = EmbeddingConstruction(self.vocab.in_word_vocab,
                                                     embedding_style['single_token_item'],
                                                     embedding_style['emb_strategy'],
                                                     hidden_size=hidden_size,
                                                     fix_word_emb=False,
                                                     fix_bert_emb=False,
                                                     device=self.device)
        self.direction_option = direction_option
        self.hidden_size = hidden_size

        if args.gnn == 'ggnn':
            self.num_layers = 2  # ggnn
            self.gnn_encoder = nn.ModuleList(
                [GGNNLayer(hidden_size, hidden_size, direction_option, num_layers=1, n_etypes=1, bias=True)
                 for i in range(self.num_layers)])
        elif args.gnn == 'gat':
            self.num_layers = 1  # gat uni/bi_fuse/bi_sep
            self.gnn_encoder = nn.ModuleList(
                [GATLayer(hidden_size, hidden_size, num_heads=1, direction_option=direction_option)
                 for i in range(self.num_layers)])
        elif args.gnn == 'graphsage':
            self.num_layers = 2  # graphsage
            self.gnn_encoder = nn.ModuleList(
                [GraphSAGELayer(hidden_size, hidden_size, aggregator_type='mean',
                                direction_option=direction_option)
                 for i in range(self.num_layers)])
        elif args.gnn == 'gcn':
            self.num_layers = 1  # gcn
            self.gnn_encoder = nn.ModuleList(
                [GCNLayer(hidden_size, hidden_size, direction_option=direction_option)
                 for i in range(self.num_layers)])

        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(hidden_size)
                                      for i in range(self.num_layers)])  # necessary for this task

        self.distmult = DistMult(rel_emb_from_gnn=False, num_relations=num_relations,
                                 embedding_dim=hidden_size, loss_name=loss_name)

        self.loss_name = loss_name
        if loss_name == 'BCELoss':
            self.loss = torch.nn.BCELoss()
        elif loss_name == "SoftplusLoss":
            self.loss = SoftplusLoss()
        elif loss_name == "SigmoidLoss":
            self.loss = SigmoidLoss()
        elif loss_name == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        elif loss_name == "MSELoss":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

        self.reset_parameters()

    def embedding(self, kg_graph):
        graph_nodes_idx = []
        node_len_list = []
        for x in kg_graph.graph_attributes['graph_nodes']:
            tmp_list = []
            for token in x.split(' '):
                tmp_list.append(self.vocab.in_word_vocab.word2index[token])

            graph_nodes_idx.append(tmp_list)
            node_len_list.append(len(tmp_list))

        max_size = max(node_len_list)
        graph_nodes_idx = [x + [self.vocab.in_word_vocab.PAD] * (max_size - len(x)) for x in graph_nodes_idx]
        graph_nodes_idx = torch.tensor(graph_nodes_idx, dtype=torch.long).to(self.device)
        node_len_tensor = torch.tensor(node_len_list, dtype=torch.long).to(self.device)
        num_nodes = torch.tensor([len(node_len_list)], dtype=torch.long).to(self.device)
        kg_graph.node_features['token_id'] = graph_nodes_idx
        node_feat = self.embedding_layer(kg_graph, node_len_tensor, num_nodes)
        return node_feat

    def reset_parameters(self):
        nn.init.xavier_normal_(self.node_emb.weight.data)

    def forward(self, kg_graph, e1, rel, e2_multi=None, require_loss=True):
        kg_graph = to_cuda(kg_graph, self.device)
        kg_graph.node_features['node_feat'] = self.embedding(kg_graph)
        list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))

        # run GNN
        # ================================================================================
        node_feats = kg_graph.node_features['node_feat']
        dgl_graph = kg_graph.to_dgl()

        if self.direction_option == 'undirected':
            node_embs = kg_graph.node_features['node_feat']
            for i in range(self.num_layers):
                if self.args.gnn == 'ggnn' or self.args.gnn == 'gcn' or self.args.gnn == 'graphsage':
                    node_embs = self.gnn_encoder[i](dgl_graph, node_embs)  # GGNN
                elif self.args.gnn == 'gat':
                    node_embs = self.gnn_encoder[i](dgl_graph, node_embs).squeeze()  # GAT
                node_embs = torch.dropout(torch.tanh(self.bn_list[i](node_embs)), 0.25, train=require_loss)
        else:
            assert node_feats.shape[1] == self.hidden_size

            zero_pad = node_feats.new_zeros((node_feats.shape[0], self.hidden_size - node_feats.shape[1]))
            node_feats = torch.cat([node_feats, zero_pad], -1)

            feat_in = node_feats
            feat_out = node_feats

            if self.direction_option == 'bi_sep':
                for i in range(self.num_layers):
                    h = self.gnn_encoder[i](dgl_graph, (feat_in, feat_out))
                    if self.args.gnn == 'ggnn' or self.args.gnn == 'graphsage' or self.args.gnn == 'gcn':
                        feat_in = torch.dropout(torch.tanh(self.bn_list[i](h[0])), 0.25, train=require_loss)
                        feat_out = torch.dropout(torch.tanh(self.bn_list[i](h[1])), 0.25, train=require_loss)
                    elif self.args.gnn == 'gat':
                        feat_in = torch.dropout(torch.tanh(self.bn_list[i](h[0].squeeze())), 0.25,
                                                train=require_loss)  # GAT
                        feat_out = torch.dropout(torch.tanh(self.bn_list[i](h[1].squeeze())), 0.25,
                                                 train=require_loss)  # GAT
                node_embs = (feat_in + feat_out) / 2
            elif self.direction_option == 'bi_fuse':
                for i in range(self.num_layers):
                    if self.args.gnn == 'ggnn':
                        h = self.gnn_encoder[i](dgl_graph, (feat_in, feat_in))  # GGNN
                        feat_in = torch.dropout(torch.tanh(self.bn_list[i](h[0])), 0.25, train=require_loss)  # GGNN
                    elif self.args.gnn == 'gat':
                        h = self.gnn_encoder[i](dgl_graph, feat_in).squeeze()  # GAT
                        feat_in = torch.dropout(torch.tanh(self.bn_list[i](h)), 0.25, train=require_loss)  # GAT
                    elif self.args.gnn == 'graphsage' or self.args.gnn == 'gcn':
                        h = self.gnn_encoder[i](dgl_graph, feat_in)  # GraphSage
                        feat_in = torch.dropout(torch.tanh(self.bn_list[i](h)), 0.25, train=require_loss)  # GraphSage
                node_embs = feat_in
            else:
                raise RuntimeError('Unknown `bidirection` value: {}'.format(self.direction_option))

        kg_graph.node_features['node_emb'] = node_embs
        # ================================================================================

        kg_graph.graph_attributes['list_e_r_pair_idx'] = list_e_r_pair_idx
        kg_graph.graph_attributes['multi_binary_label'] = e2_multi

        # down-task
        kg_graph = self.distmult(kg_graph)
        if require_loss:
            if self.loss_name == "SoftplusLoss" or self.loss_name == "SigmoidLoss":
                loss = self.loss(kg_graph.graph_attributes['p_score'],
                                 kg_graph.graph_attributes['n_score'])
            else:
                loss = self.loss(kg_graph.graph_attributes['logits'], e2_multi)
            return kg_graph.graph_attributes['logits'], loss
        else:
            return kg_graph.graph_attributes['logits']


class Kinship:
    def __init__(self, device, args):
        super(Kinship, self).__init__()
        self.device = device
        self._build_dataloader()
        self._build_model(args)
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = KinshipDataset(root_dir='/raid/ghn/graph4nlp/examples/pytorch/kg_completion/kinship',
                                 topology_builder=None,
                                 topology_subdir='e1rel_to_e2')

        # TODO: When edge2node is True, num_entities != train_set.KG_graph.get_node_num()
        self.num_entities = dataset.KG_graph.graph_attributes['num_entities']
        self.num_relations = dataset.KG_graph.graph_attributes['num_relations']
        self.kg_graph = dataset.KG_graph

        self.train_dataloader = DataLoader(dataset.train, batch_size=64, shuffle=True,
                                           num_workers=1,
                                           collate_fn=dataset.collate_fn)

        self.val_dataloader = DataLoader(dataset.val, batch_size=64, shuffle=True,
                                         num_workers=1,
                                         collate_fn=dataset.collate_fn)

        self.test_dataloader = DataLoader(dataset.test, batch_size=64, shuffle=True,
                                          num_workers=1,
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model

    def _build_model(self, args):  # BCELoss SigmoidLoss
        self.model = Graph2DistMult(args,
                                    self.vocab,
                                    self.device,
                                    num_entities=self.num_entities,
                                    num_relations=self.num_relations,
                                    loss_name=args.loss,
                                    direction_option=args.direction_option).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=2e-3)

    def _build_evaluation(self):
        self.metrics = [RankingAndHits(
            model_path='best_graph2distmult_' + self.model.direction_option + '_' + self.model.loss_name)]

    def train(self):
        for epoch in range(30):
            self.model.train()
            loss_list = []

            for data in self.train_dataloader:
                e1, rel, e2_multi, e2_multi_tensor_idx, \
                e2, rel_eval, e1_multi, e1_multi_tensor_idx = data

                e1 = torch.cat([e1, e2]).to(self.device)
                rel = torch.cat([rel, rel_eval]).to(self.device)
                e2_multi = torch.cat([e2_multi, e1_multi], dim=0).to(self.device)
                # e2_multi_tensor_idx = torch.cat([e2_multi_tensor_idx, e1_multi_tensor_idx]).to(self.device)

                # TODO
                if random.randint(0, 1) == 0:
                    e1 = torch.cat([e1]).to(self.device)
                    rel = torch.cat([rel]).to(self.device)
                    e2_multi = torch.cat([e2_multi], dim=0).to(self.device)
                else:
                    e1 = torch.cat([e2]).to(self.device)
                    rel = torch.cat([rel_eval]).to(self.device)
                    e2_multi = torch.cat([e1_multi], dim=0).to(self.device)

                # c = torch.randperm(e1.size()[0])
                # e1 = e1[c]
                # rel = rel[c]
                # e2_multi = e2_multi[c]

                # e1 = e1.to(self.device)
                # rel = rel.to(self.device)
                # e2_multi = e2_multi.to(self.device)

                _, loss = self.model(self.kg_graph, e1, rel, e2_multi, require_loss=True)
                loss_list.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('#' * 50)
            print("Epoch: {}".format(epoch))
            print('train loss =' + str(sum(loss_list) / len(loss_list)))
            print('#' * 50)
            self.valid()

        return self.metrics[0].best_mrr

    def valid(self):
        self.model.eval()

        self.metrics[0].calculate_scores(self.model,
                                         self.val_dataloader,
                                         name='validation',
                                         kg_graph=self.kg_graph,
                                         device=self.device)

        return self.metrics[0].best_mrr

    def test(self):
        print('load best model...')
        self.model.load_state_dict(torch.load(self.metrics[0].model_path))
        self.model.eval()
        self.metrics[0].calculate_scores(self.model,
                                         self.test_dataloader,
                                         name='test',
                                         kg_graph=self.kg_graph,
                                         device=self.device,
                                         save=False)

        return self.metrics[0].best_mrr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--direction_option', type=str, default='undirected')
    parser.add_argument('--loss', type=str, default='SigmoidLoss')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('using cuda...')
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    runner = Kinship(device, args)
    max_score = runner.train()
    max_score = runner.test()
    print("Train finish, best MRR: {:.3f}".format(max_score))

# nohup python -m examples.pytorch.kg_completion.kinship.main >> distmult_bce_gat_uni.log 2>&1 &