import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# os.environ['CUDA_LAUNCH_BLOCKING'] = "5"
from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.datasets.kinship import KinshipDataset, KinshipTestDataset
# from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
# from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
# from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
from graph4nlp.pytorch.modules.prediction.classification.kg_completion.DistMult import DistMult, DistMultLayer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import dgl
import numpy as np

from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import EmbeddingConstruction
from graph4nlp.pytorch.modules.loss.kg_loss import *

class RankingAndHits(EvaluationMetricBase):
    def __init__(self, model_path='best_graph2distmult', batch_size=128):
        super(RankingAndHits, self).__init__()
        self.batch_size = batch_size
        self.best_mrr = 0.
        self.best_hits1 = 0.
        self.best_hits10 = 0.
        self.model_path = model_path

    def calculate_scores(self, model, dataloader, name, kg_graph, device):
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
                # if model.loss_name == "SoftplusLoss" or model.loss_name == "SigmoidLoss":
                #     pred1, pos1, neg1 = model(kg_graph, e1, rel, e2_multi, require_loss=False)
                #     pred2, pos2, neg2 = model(kg_graph, e2, rel_eval, e1_multi, require_loss=False)
                # else:
                pred1 = model(kg_graph, e1, rel, e2_multi, require_loss=False)
                pred2 = model(kg_graph, e2, rel_eval, e1_multi, require_loss=False)
                pred1, pred2 = pred1.data, pred2.data
                e1, e2 = e1.data, e2.data
                e2_multi1, e2_multi2 = e2_multi_tensor_idx.data, e1_multi_tensor_idx.data

                if e1.size()[0]!=self.batch_size:
                    continue

                # if model.loss_name == "SoftplusLoss" or model.loss_name == "SigmoidLoss":
                #     e2_multi1_binary = str2var['e2_multi1_binary'].float()
                #     e2_multi2_binary = str2var['e2_multi2_binary'].float()
                #     pred1, pos1, neg1 = model.forward(e1, rel, X, adjacencies, e2_multi1_binary)
                #     pred2, pos2, neg2 = model.forward(e2, rel_reverse, X, adjacencies, e2_multi2_binary)
                # else:
                #     pred1 = model.forward(e1, rel, X, adjacencies)
                #     pred2 = model.forward(e2, rel_reverse, X, adjacencies)
                # pred1, pred2 = pred1.data, pred2.data
                # e1, e2 = e1.data, e2.data
                # e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
                for i in range(self.batch_size):
                    # these filters contain ALL labels
                    filter1 = e2_multi1[i].long()
                    filter2 = e2_multi2[i].long()

                    num = e1[i, 0].item()
                    # save the prediction that is relevant
                    target_value1 = pred1[i,e2.cpu().numpy()[i, 0].item()].item()
                    target_value2 = pred2[i,e1.cpu().numpy()[i, 0].item()].item()
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
                    rank1 = np.where(argsort1[i]==e2.cpu().numpy()[i, 0])[0][0]
                    if model.loss_name in ['SoftplusLoss', 'SigmoidLoss'] and max_values1[i][rank1] == max_values1[i][0]:
                        rank1 = 0
                    rank2 = np.where(argsort2[i]==e1.cpu().numpy()[i, 0])[0][0]
                    if model.loss_name in ['SoftplusLoss', 'SigmoidLoss'] and max_values2[i][rank2] == max_values2[i][0]:
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

        if np.mean(hits[0])>self.best_hits1:
            self.best_hits1 = np.mean(hits[0])

        if np.mean(hits[9])>self.best_hits10:
            self.best_hits10 = np.mean(hits[9])

        if np.mean(1. / np.array(ranks))>self.best_mrr:
            self.best_mrr = np.mean(1. / np.array(ranks))
            print('saving best model...')
            torch.save(model.state_dict(), self.model_path)

        print('-' * 50)
        print('')

# TODO: 1. initialize graph.node_features['edge_feat']/self.distmult.rel_emb with self.embedding_layer
# TODO: 2. learn graph.node_features['edge_emb'] from GNN (edge2node)

class Graph2DistMult(nn.Module):
    def __init__(self, vocab, num_entities, hidden_size=300, num_relations=None, direction_option='uni', loss_name='BCELoss'):
        super(Graph2DistMult, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.vocab = vocab
        self.num_entities = num_entities

        self.node_emb = nn.Embedding(num_entities, hidden_size)

        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "none"}

        self.embedding_layer = EmbeddingConstruction(self.vocab.in_word_vocab,
                                                     embedding_style['word_emb_type'],
                                                     embedding_style['node_edge_emb_strategy'],
                                                     embedding_style['seq_info_encode_strategy'],
                                                     hidden_size=hidden_size,
                                                     fix_word_emb=False,
                                                     dropout=0.2,
                                                     device=self.device)

        # self.graph_topology = IEBasedGraphConstruction(vocab=self.vocab.in_word_vocab,
        #                                                embedding_style=embedding_style,
        #                                                use_cuda=False)
        # output_graph0 = iegc.embedding(output_graph0)


        # self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
        #                                                        vocab=vocab.in_word_vocab,
        #                                                        hidden_size=hidden_size, dropout=0.2, use_cuda=True,
        #                                                        fix_word_emb=False)

        # self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer

        # self.gnn_encoder = GAT(2, hidden_size, hidden_size, hidden_size, [2, 1], direction_option=direction_option)
        self.gnn_encoder = GGNN(1, hidden_size, hidden_size, direction_option=direction_option)
        self.bn = torch.nn.BatchNorm1d(hidden_size)  # necessary for this task

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

    # def embedding(self, graph: GraphData):
    #     node_attributes = graph.node_attributes
    #     edge_attributes = graph.edge_attributes
    #
    #     # Build embedding(initial feature vector) for graph nodes.
    #     # Each node may contains multiple tokens.
    #     node_idxs_list = []
    #     node_len_list = []
    #     for node_id, node_dict in node_attributes.items():
    #         node_word_idxs = []
    #         for token in node_dict['token'].split():
    #             node_word_idxs.append(self.vocab.in_word_vocab.getIndex(token))
    #         node_idxs_list.append(node_word_idxs)
    #         node_len_list.append(len(node_word_idxs))
    #     max_size = max(node_len_list)
    #     node_idxs_list = [x+[self.vocab.in_word_vocab.PAD]*(max_size-len(x)) for x in node_idxs_list]
    #     node_idxs_tensor = torch.LongTensor(node_idxs_list).to(self.device)
    #     # if self.embedding_layer.node_edge_emb_strategy == 'mean':
    #     #     node_len_tensor = torch.LongTensor(node_len_list).view(-1, 1)
    #     # else:
    #     node_len_tensor = torch.LongTensor(node_len_list).to(self.device)
    #     num_nodes = torch.LongTensor([len(node_len_list)]).to(self.device)
    #     node_feat = self.embedding_layer(node_idxs_tensor, node_len_tensor, num_nodes)
    #     graph.node_features['node_feat'] = node_feat
    #
    #     if 'token' in edge_attributes[0].keys():
    #         # If edge information is stored in `edge_attributes`,
    #         # build embedding(initial feature vector) for graph edges.
    #         # Each edge may contains multiple tokens.
    #         edge_idxs_list = []
    #         edge_len_list = []
    #         for edge_id, edge_dict in edge_attributes.items():
    #             edge_word_idxs = []
    #             for token in edge_dict['token'].split():
    #                 edge_word_idxs.append(self.vocab.in_word_vocab.getIndex(token))
    #             edge_idxs_list.append(edge_word_idxs)
    #             edge_len_list.append(len(edge_word_idxs))
    #
    #         max_size = max(edge_len_list)
    #         edge_idxs_list = [x + [self.vocab.in_word_vocab.PAD] * (max_size - len(x)) for x in edge_idxs_list]
    #         edge_idxs_tensor = torch.LongTensor(edge_idxs_list).to(self.device)
    #         # if self.embedding_layer.node_edge_emb_strategy == 'mean':
    #         #     edge_len_tensor = torch.LongTensor(edge_len_list).view(-1, 1)
    #         # else:
    #         edge_len_tensor = torch.LongTensor(edge_len_list).to(self.device)
    #         num_edges = torch.LongTensor([len(edge_len_list)]).to(self.device)
    #         edge_feat = self.embedding_layer(edge_idxs_tensor, edge_len_tensor, num_edges)
    #         graph.edge_features['edge_feat'] = edge_feat
    #
    #     return graph

    def embedding(self, kg_graph):
        # graph_nodes_idx = [self.vocab.in_word_vocab.word2index[token]
        #                    for x in kg_graph.graph_attributes['graph_nodes']
        #                    for token in x.split(' ')]
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
        node_feat = self.embedding_layer(graph_nodes_idx, node_len_tensor, num_nodes)
        return node_feat

    def reset_parameters(self):
        nn.init.xavier_normal_(self.node_emb.weight.data)
        # xavier_normal_(self.emb_rel.weight.data)

    def forward(self, kg_graph, e1, rel, e2_multi=None, require_loss=True):
        # kg_graph.node_features['node_feat'] = self.node_emb.weight

        # kg_graph.node_features['node_emb'] = self.node_emb.weight
        kg_graph.node_features['node_feat'] = self.embedding(kg_graph)

        list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))

        # run GNN
        kg_graph = self.gnn_encoder(kg_graph)
        kg_graph.node_features['node_emb'] = self.bn(kg_graph.node_features['node_emb'])
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
    def __init__(self):
        super(Kinship, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        # train_set = KinshipDataset(root_dir="graph4nlp/pytorch/test/dataset/kinship",
        # train_set = KinshipDataset(root_dir="/mnt/graph4nlp/graph4nlp/pytorch/test/dataset/kinship",
        train_set = KinshipDataset(root_dir="/Users/gaohanning/PycharmProjects/graph4nlp/graph4nlp/pytorch/test/dataset/kinship",
                                  topology_builder=None,
                                  topology_subdir='e1rel_to_e2',
                                  share_vocab=True)

        # TODO: When edge2node is True, num_entities != train_set.KG_graph.get_node_num()
        self.num_entities = train_set.KG_graph.graph_attributes['num_entities']
        self.num_relations = train_set.KG_graph.graph_attributes['num_relations']

        self.kg_graph = train_set.KG_graph

        test_set = KinshipTestDataset(
            root_dir="/Users/gaohanning/PycharmProjects/graph4nlp/graph4nlp/pytorch/test/dataset/kinship",
            # root_dir="/mnt/graph4nlp/graph4nlp/pytorch/test/dataset/kinship",
            topology_builder=None,
            topology_subdir='e1rel_to_e2',
            share_vocab=True,
            KG_graph=self.kg_graph)

        self.train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True,
                                           num_workers=1,
                                           collate_fn=train_set.collate_fn)
        self.test_dataloader = DataLoader(test_set, batch_size=128, shuffle=True,
                                          num_workers=1,
                                          collate_fn=test_set.collate_fn)
        self.vocab = train_set.vocab_model

    def _build_model(self):
        self.model = Graph2DistMult(self.vocab,
                                    num_entities=self.num_entities,
                                    num_relations=self.num_relations,
                                    loss_name='SigmoidLoss').to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=1e-3)

    def _build_evaluation(self):
        self.metrics = [RankingAndHits()]

    def train(self):
        for epoch in range(200):
            self.model.train()
            loss_list = []

            for data in self.train_dataloader:
                e1, rel, e2_multi, e2_multi_tensor_idx = data
                e1 = e1.to(self.device)
                rel = rel.to(self.device)
                e2_multi = e2_multi.to(self.device)
                _, loss = self.model(self.kg_graph, e1, rel, e2_multi, require_loss=True)
                loss_list.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('#' * 50)
            print("Epoch: {}".format(epoch))
            print('train loss =' + str(sum(loss_list)/len(loss_list)))
            print('#' * 50)
            self.evaluate()

    def evaluate(self):
        self.model.eval()

        self.metrics[0].calculate_scores(self.model,
                                         self.test_dataloader,
                                         name='test',
                                         kg_graph=self.kg_graph,
                                         device=self.device)


if __name__ == "__main__":
    runner = Kinship()
    max_score = runner.train()
    print("Train finish, best score: {:.3f}".format(max_score))

# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = "5"
# from graph4nlp.pytorch.data.data import GraphData
# from graph4nlp.pytorch.datasets.kinship import KinshipDataset, KinshipTestDataset
# # from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
# # from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
# # from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
# from graph4nlp.pytorch.modules.graph_construction.gat import GAT
# from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
# from graph4nlp.pytorch.modules.prediction.classification.kg_completion.DistMult import DistMult
#
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import dgl
#
# from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
# from graph4nlp.pytorch.modules.graph_construction.embedding_construction import EmbeddingConstruction
# from graph4nlp.pytorch.modules.loss.kg_loss import *
#
# class ExpressionAccuracy(EvaluationMetricBase):
#     def __init__(self):
#         super(ExpressionAccuracy, self).__init__()
#
#     def calculate_scores(self, ground_truth, predict):
#         correct = 0
#         assert len(ground_truth) == len(predict)
#         for gt, pred in zip(ground_truth, predict):
#             print("ground truth: ", gt)
#             print("prediction: ", pred)
#
#             if gt == pred:
#                 correct += 1.
#         return correct / len(ground_truth)
#
#
# def logits2seq(prob: torch.Tensor):
#     ids = prob.argmax(dim=-1)
#     return ids
#
#
# def wordid2str(word_ids, vocab: Vocab):
#     ret = []
#     assert len(word_ids.shape) == 2, print(word_ids.shape)
#     for i in range(word_ids.shape[0]):
#         id_list = word_ids[i, :]
#         ret_inst = []
#         for j in range(id_list.shape[0]):
#             if id_list[j] == vocab.EOS:
#                 break
#             token = vocab.getWord(id_list[j])
#             ret_inst.append(token)
#         ret.append(" ".join(ret_inst))
#     return ret
#
#
# class Graph2seqLoss(nn.Module):
#     def __init__(self, vocab: Vocab):
#         super(Graph2seqLoss, self).__init__()
#         self.loss_func = nn.NLLLoss()
#         self.VERY_SMALL_NUMBER = 1e-31
#         self.vocab = vocab
#
#     def forward(self, prob, gt):
#         assert prob.shape[0:1] == gt.shape[0:1]
#         assert len(prob.shape) == 3
#         log_prob = torch.log(prob + self.VERY_SMALL_NUMBER)
#         batch_size = gt.shape[0]
#         step = gt.shape[1]
#
#         mask = 1 - gt.data.eq(self.vocab.PAD).float()
#
#         prob_select = torch.gather(log_prob.view(batch_size * step, -1), 1, gt.view(-1, 1))
#
#         prob_select_masked = - torch.masked_select(prob_select, mask.view(-1, 1).byte())
#         loss = torch.mean(prob_select_masked)
#         return loss
#
#
# class Graph2DistMult(nn.Module):
#     def __init__(self, vocab, hidden_size=300, num_relations=None, direction_option='bi_fuse', loss_name='BCELoss'):
#         super(Graph2DistMult, self).__init__()
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.vocab = vocab
#         embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
#                            'seq_info_encode_strategy': "none"}
#
#         self.embedding_layer = EmbeddingConstruction(self.vocab.in_word_vocab,
#                                                      embedding_style['word_emb_type'],
#                                                      embedding_style['node_edge_emb_strategy'],
#                                                      embedding_style['seq_info_encode_strategy'],
#                                                      hidden_size=hidden_size,
#                                                      fix_word_emb=False,
#                                                      dropout=0.2,
#                                                      device=self.device)
#
#         # self.graph_topology = IEBasedGraphConstruction(vocab=self.vocab.in_word_vocab,
#         #                                                embedding_style=embedding_style,
#         #                                                use_cuda=False)
#         # output_graph0 = iegc.embedding(output_graph0)
#
#
#         # self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
#         #                                                        vocab=vocab.in_word_vocab,
#         #                                                        hidden_size=hidden_size, dropout=0.2, use_cuda=True,
#         #                                                        fix_word_emb=False)
#
#         # self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer
#         self.gnn_encoder = GAT(2, hidden_size, hidden_size, hidden_size, [2, 1], direction_option=direction_option)
#         self.distmult = DistMult(rel_emb_from_gnn=False, num_relations=num_relations, embedding_dim=hidden_size)
#         # self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)
#
#         self.loss_name = loss_name
#         if loss_name == 'BCELoss':
#             self.loss = torch.nn.BCELoss()
#         elif loss_name == "SoftplusLoss":
#             self.loss = SoftplusLoss()
#         elif loss_name == "SigmoidLoss":
#             self.loss = SigmoidLoss()
#         elif loss_name == "SoftMarginLoss":
#             self.loss = nn.SoftMarginLoss()
#         elif loss_name == "MSELoss":
#             self.loss = nn.MSELoss()
#         else:
#             raise NotImplementedError()
#
#     def embedding(self, graph: GraphData):
#         node_attributes = graph.node_attributes
#         edge_attributes = graph.edge_attributes
#
#         # Build embedding(initial feature vector) for graph nodes.
#         # Each node may contains multiple tokens.
#         node_idxs_list = []
#         node_len_list = []
#         for node_id, node_dict in node_attributes.items():
#             node_word_idxs = []
#             for token in node_dict['token'].split():
#                 node_word_idxs.append(self.vocab.in_word_vocab.getIndex(token))
#             node_idxs_list.append(node_word_idxs)
#             node_len_list.append(len(node_word_idxs))
#         max_size = max(node_len_list)
#         node_idxs_list = [x+[self.vocab.in_word_vocab.PAD]*(max_size-len(x)) for x in node_idxs_list]
#         node_idxs_tensor = torch.LongTensor(node_idxs_list)
#         # if self.embedding_layer.node_edge_emb_strategy == 'mean':
#         #     node_len_tensor = torch.LongTensor(node_len_list).view(-1, 1)
#         # else:
#         node_len_tensor = torch.LongTensor(node_len_list)
#         num_nodes = torch.LongTensor([len(node_len_list)])
#         node_feat = self.embedding_layer(node_idxs_tensor, node_len_tensor, num_nodes)
#         graph.node_features['node_feat'] = node_feat
#
#         if 'token' in edge_attributes[0].keys():
#             # If edge information is stored in `edge_attributes`,
#             # build embedding(initial feature vector) for graph edges.
#             # Each edge may contains multiple tokens.
#             edge_idxs_list = []
#             edge_len_list = []
#             for edge_id, edge_dict in edge_attributes.items():
#                 edge_word_idxs = []
#                 for token in edge_dict['token']:
#                     edge_word_idxs.append(self.vocab.in_word_vocab.getIndex(token))
#                 edge_idxs_list.append(edge_word_idxs)
#                 edge_len_list.append(len(edge_word_idxs))
#
#             max_size = max(edge_len_list)
#             edge_idxs_list = [x + [self.vocab.in_word_vocab.PAD] * (max_size - len(x)) for x in edge_idxs_list]
#             edge_idxs_tensor = torch.LongTensor(edge_idxs_list)
#             # if self.embedding_layer.node_edge_emb_strategy == 'mean':
#             #     edge_len_tensor = torch.LongTensor(edge_len_list).view(-1, 1)
#             # else:
#             edge_len_tensor = torch.LongTensor(edge_len_list)
#             num_edges = torch.LongTensor([len(edge_len_list)])
#             edge_feat = self.embedding_layer(edge_idxs_tensor, edge_len_tensor, num_edges)
#             graph.edge_features['edge_feat'] = edge_feat
#
#         return graph
#
#     def forward(self, kg_graph, e1, rel, e2_multi=None, require_loss=True):
#         kg_graph = self.embedding(kg_graph)
#         # batch_dgl_graph = self.graph_topology(graph_list)
#         # do graph nn here
#         # convert DGLGraph to GraphData
#         # batch_graph = GraphData()
#         # batch_graph.from_dgl(kg_graph)
#         # kg_graph_dgl = kg_graph.to_dgl()
#
#         list_e_r_pair_idx = list(zip(e1.squeeze().tolist(), rel.squeeze().tolist()))
#
#         # run GNN
#         batch_graph = self.gnn_encoder(kg_graph)
#         batch_graph.graph_attributes['list_e_r_pair_idx'] = list_e_r_pair_idx
#         # kg_graph_dgl.ndata['node_emb'] = kg_graph.node_features['node_emb']
#         # kg_graph_dgl.ndata['rnn_emb'] = kg_graph.node_features['node_feat']
#
#         # dgl_graph_list = dgl.unbatch(kg_graph_dgl)
#         # for g, dg in zip(graph_list, dgl_graph_list):
#         #     g.node_features["node_emb"] = dg.ndata["node_emb"]
#         #     g.node_features["rnn_emb"] = dg.ndata["rnn_emb"]
#
#         # down-task
#         batch_graph = self.distmult(batch_graph)
#         if require_loss:
#             loss = self.loss(batch_graph.graph_attributes['logits'], e2_multi)
#             return batch_graph.graph_attributes['logits'], loss
#         else:
#             return batch_graph.graph_attributes['logits']
#
#
# class Kinship:
#     def __init__(self):
#         super(Kinship, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self._build_dataloader()
#         self._build_model()
#         self._build_optimizer()
#         self._build_evaluation()
#
#     def _build_dataloader(self):
#         # train_set = KinshipDataset(root_dir="graph4nlp/pytorch/test/dataset/kinship",
#         train_set = KinshipDataset(root_dir="/Users/gaohanning/PycharmProjects/graph4nlp/graph4nlp/pytorch/test/dataset/kinship",
#                                   topology_builder=None,
#                                   topology_subdir='e1rel_to_e2',
#                                   share_vocab=True)
#
#         # TODO: When edge2node is True, num_entities != train_set.KG_graph.get_node_num()
#         self.num_entities = train_set.KG_graph.graph_attributes['num_entities']
#         self.num_relations = train_set.KG_graph.graph_attributes['num_relations']
#
#         self.kg_graph = train_set.KG_graph
#
#         test_set = KinshipTestDataset(
#             root_dir="/Users/gaohanning/PycharmProjects/graph4nlp/graph4nlp/pytorch/test/dataset/kinship",
#             topology_builder=None,
#             topology_subdir='e1rel_to_e2',
#             share_vocab=True)
#
#         self.train_dataloader = DataLoader(train_set, batch_size=24, shuffle=True,
#                                            num_workers=1,
#                                            collate_fn=train_set.collate_fn)
#         self.test_dataloader = DataLoader(test_set, batch_size=24, shuffle=True,
#                                           num_workers=1,
#                                           collate_fn=test_set.collate_fn)
#         self.vocab = train_set.vocab_model
#
#     def _build_model(self):
#         self.model = Graph2DistMult(self.vocab, num_relations=self.num_relations).to(self.device)
#
#     def _build_optimizer(self):
#         parameters = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = optim.Adam(parameters, lr=1e-3)
#
#     def _build_evaluation(self):
#         self.metrics = [ExpressionAccuracy()]
#
#     def train(self):
#         max_score = -1
#         for epoch in range(200):
#             self.model.train()
#             print("Epoch: {}".format(epoch))
#             for data in self.train_dataloader:
#                 e1, rel, e2_multi = data
#                 e1 = e1.to(self.device)
#                 rel = rel.to(self.device)
#                 e2_multi = e2_multi.to(self.device)
#                 _, loss = self.model(self.kg_graph, e1, rel, e2_multi, require_loss=True)
#                 print(loss.item())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 # self.evaluate()
#             if epoch > 4:
#                 self.evaluate()
#                 # score = self.evaluate()
#                 # max_score = max(max_score, score)
#         return max_score
#
#     def evaluate(self):
#         self.model.eval()
#         pred_collect = []
#         gt_collect = []
#         acc_list = []
#         for data in self.test_dataloader:
#             e1, rel, e2_multi = data
#             e1 = e1.to(self.device)
#             rel = rel.to(self.device)
#             e2_multi = e2_multi.to(self.device)
#
#             prob = self.model(self.kg_graph, e1, rel, e2_multi, require_loss=False)
#             acc = torch.sum(torch.gt(prob, 0.5)==e2_multi).item()/(self.test_dataloader.batch_size*self.kg_graph.graph_attributes['num_entities'])
#             # print('acc='+str(acc))
#             # break
#             acc_list.append(acc)
#         print('acc='+str(sum(acc_list)/len(acc_list)))
#             # pred = logits2seq(prob)
#             #
#             # pred_str = wordid2str(pred.detach().cpu(), self.vocab.in_word_vocab)
#             # tgt_str = wordid2str(tgt, self.vocab.in_word_vocab)
#             # pred_collect.extend(pred_str)
#             # gt_collect.extend(tgt_str)
#
#         # score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
#         # print("accuracy: {:.3f}".format(score))
#         # return score
#
#
# if __name__ == "__main__":
#     runner = Kinship()
#     max_score = runner.train()
#     print("Train finish, best score: {:.3f}".format(max_score))
