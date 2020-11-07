import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import os
import json
import time
import argparse
import random
from decimal import Decimal
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from graph4nlp.pytorch.data.data import *
# from .semiEval import SemiEvalDataset
from .dataset import SemiEvalDataset
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN
from .dependency_graph_construction_without_tokenize import DependencyBasedGraphConstruction_without_tokenizer
from .line_graph_construction import LineBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import *
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import *
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import WordEmbedding
from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding.gcn import GCN
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import dgl

from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
# from graph4nlp.pytorch.modules.prediction.classification.node_classification.FeedForwardNN import FeedForwardNN
# from graph4nlp.pytorch.modules.prediction.classification.node_classification.BiLSTMFeedForwardNN import BiLSTMFeedForwardNN
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy



def logits2tag(logits):
        _, pred=torch.max(logits,dim=-1)
        #print(pred.size())
        return pred     

        
        

         
                   
class Word2tag(nn.Module):
    def __init__(self, vocab, device=None):
        super(Word2tag, self).__init__()
        self.vocab = vocab
        self.device =device
        
        embedding_style = {'single_token_item': True if args.graph_type != 'ie' else False,
                            'emb_strategy': 'w2v_bilstm',
                            'num_rnn_layers': 1,
                            'bert_model_name': 'bert-base-uncased',
                            'bert_lower_case': True
                           }
        use_edge_weight = False
        if args.graph_type=='line_graph':
          if args.gnn_type=='ggnn':  
              self.graph_topology = LineBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=int(args.init_hidden_size/2), 
                                                               rnn_dropout=args.rnn_dropout,
                                                               word_dropout=args.word_dropout,
                                                               device=self.device,
                                                               fix_word_emb=not args.no_fix_word_emb,
                                                               fix_bert_emb=not args.no_fix_bert_emb)
          else:
              self.graph_topology = LineBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=args.init_hidden_size, 
                                                               rnn_dropout=args.rnn_dropout,
                                                               word_dropout=args.word_dropout, 
                                                               device=self.device,
                                                               fix_word_emb=not args.no_fix_word_emb,
                                                               fix_bert_emb=not args.no_fix_bert_emb)              
        if args.graph_type=='dependency_graph':
          if args.gnn_type=='ggnn': 
              self.graph_topology = DependencyBasedGraphConstruction_without_tokenizer(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=int(args.init_hidden_size/2), 
                                                               rnn_dropout=args.rnn_dropout,
                                                               word_dropout=args.word_dropout, 
                                                               device=self.device,
                                                               fix_word_emb=not args.no_fix_word_emb,
                                                               fix_bert_emb=not args.no_fix_bert_emb)
          else:
              self.graph_topology = DependencyBasedGraphConstruction_without_tokenizer(embedding_style=embedding_style,
                                                   vocab=vocab.in_word_vocab,
                                                   hidden_size=args.init_hidden_size, 
                                                   rnn_dropout=args.rnn_dropout,
                                                   word_dropout=args.word_dropout, 
                                                   device=self.device,
                                                   fix_word_emb=not args.no_fix_word_emb,
                                                   fix_bert_emb=not args.no_fix_bert_emb) 
        if args.graph_type=='node_emb':
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                                    vocab.in_word_vocab,
                                    embedding_style,
                                    sim_metric_type=args.gl_metric_type,
                                    num_heads=args.gl_num_heads,
                                    top_k_neigh=args.gl_top_k,
                                    epsilon_neigh=args.gl_epsilon,
                                    smoothness_ratio=args.gl_smoothness_ratio,
                                    connectivity_ratio=args.gl_connectivity_ratio,
                                    sparsity_ratio=args.gl_sparsity_ratio,
                                    input_size=args.init_hidden_size,
                                    hidden_size=args.init_hidden_size,
                                    fix_word_emb=not args.no_fix_word_emb,
                                    fix_bert_emb=not args.no_fix_bert_emb,
                                    word_dropout=args.word_dropout,
                                    rnn_dropout=None,
                                    device=self.device)
            use_edge_weight = True  
            
        if args.graph_type=='node_emb_refined':        
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                                    vocab.in_word_vocab,
                                    embedding_style,
                                    args.init_adj_alpha,
                                    sim_metric_type=args.gl_metric_type,
                                    num_heads=args.gl_num_heads,
                                    top_k_neigh=args.gl_top_k,
                                    epsilon_neigh=args.gl_epsilon,
                                    smoothness_ratio=args.gl_smoothness_ratio,
                                    connectivity_ratio=args.gl_connectivity_ratio,
                                    sparsity_ratio=args.gl_sparsity_ratio,
                                    input_size=args.init_hidden_size,
                                    hidden_size=args.init_hidden_size,
                                    fix_word_emb=not args.no_fix_word_emb,
                                    word_dropout=args.word_dropout,
                                    rnn_dropout=None,
                                    device=self.device)
            use_edge_weight = True        
        
        if 'w2v' in self.graph_topology.embedding_layer.word_emb_layers:
            self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                            self.vocab.in_word_vocab.embeddings.shape[0],
                            self.vocab.in_word_vocab.embeddings.shape[1],
                            pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                            fix_emb=not args.no_fix_word_emb,
                            device=self.device).word_emb_layer      
            
        self.gnn_type=args.gnn_type
        #self.linear0=nn.Linear(int(args.hidden_size*2), args.hidden_size).to(self.device)
        self.linear1=nn.Linear(int(args.hidden_size*3), args.num_class).to(self.device)
        self.dropout_gnn = nn.Dropout(args.gnn_dropout)
        self.dropout_rnn_out = nn.Dropout(p=args.rnn_dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
   
        if self.gnn_type=="graphsage":   
               if args.direction_option=='bi_sep':
                   self.gnn = GraphSAGE(args.gnn_num_layers, args.init_hidden_size, int(args.hidden_size/2), int(args.hidden_size/2), aggregator_type='mean',direction_option=args.direction_option, activation=F.elu).to(self.device)        
               else:                   
                   self.gnn = GraphSAGE(args.gnn_num_layers, args.init_hidden_size, args.hidden_size, args.hidden_size, aggregator_type='mean',direction_option=args.direction_option, activation=F.elu).to(self.device)                       
        elif self.gnn_type=="ggnn":
               if args.direction_option=='bi_sep':
                  self.gnn = GGNN(args.gnn_num_layers, int(args.init_hidden_size/2), int(args.hidden_size/2),direction_option=args.direction_option,n_etypes=1).to(self.device)        
               else:    
                  self.gnn = GGNN(args.gnn_num_layers, args.init_hidden_size, args.hidden_size,direction_option=args.direction_option,n_etypes=1).to(self.device)                       
        elif self.gnn_type=="gat":
               heads = 2
               if args.direction_option=='bi_sep': 
                   self.gnn = GAT(args.gnn_num_layers,args.init_hidden_size,int(args.hidden_size/2),int(args.hidden_size/2), heads,direction_option=args.direction_option,feat_drop=0.6, attn_drop=0.6, negative_slope=0.2, activation=F.elu).to(self.device)                                    
               else:
                   self.gnn = GAT(args.gnn_num_layers,args.init_hidden_size,args.hidden_size,args.hidden_size, heads,direction_option=args.direction_option,feat_drop=0.6, attn_drop=0.6, negative_slope=0.2, activation=F.elu).to(self.device)                       
        elif self.gnn_type=="gcn":   
               if args.direction_option=='bi_sep':
                   self.gnn = GCN(args.gnn_num_layers, args.init_hidden_size, int(args.hidden_size/2), int(args.hidden_size/2), direction_option=args.direction_option, activation=F.elu).to(self.device)        
               else:                   
                   self.gnn = GCN(args.gnn_num_layers, args.init_hidden_size, args.hidden_size, args.hidden_size,direction_option=args.direction_option, activation=F.elu).to(self.device)

     

    def forward(self, graph_list, tgt=None, pos_info=None, require_loss=True):
        batch_graph = self.graph_topology(graph_list)
        
        # run GNN
        if self.gnn_type=="ggnn":
            batch_graph.node_features['node_feat']=batch_graph.node_features['node_feat']
        else:
            batch_graph.node_features['node_feat']=self.dropout_rnn_out(batch_graph.node_features['node_feat'])

        batch_graph = self.gnn(batch_graph)
        batch_graph.node_features['node_emb'] = self.dropout_gnn(batch_graph.node_features['node_emb'])
        # batch_graph.node_features['node_emb'] = batch_graph.node_features['node_feat']

        # down-task
        graph_l = from_batch(batch_graph)
        h = []
        for idx, graph in enumerate(graph_l):
            pos = pos_info[idx]
            pos_e1, pos_e2 = pos
            sent = torch.max(graph.node_features['node_emb'], dim=0)[0]
            e1 = torch.max(graph.node_features['node_emb'][pos_e1], dim=0)[0]
            e2 = torch.max(graph.node_features['node_emb'][pos_e2], dim=0)[0]
            h.append(torch.cat([sent, e1, e2]).reshape(1, -1))

        h = torch.cat(h, dim=0)

        # logits=self.linear1(F.elu(self.linear0(h)))
        logits = self.linear1(h)

        loss = self.loss(logits.to(self.device), torch.tensor(tgt).to(self.device))
        # probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        pred = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        loss += 0.003 * (h ** 2).sum(1).mean()
        
        return pred, loss
        

class SemiEval:
    def __init__(self):
        super(SemiEval, self).__init__()
        if args.gpu>-1:
            self.device = torch.device('cuda') 
        else:
             self.device = torch.device('cpu')
        self.checkpoint_path='./checkpoints/'
        if not os.path.exists(self.checkpoint_path):
           os.mkdir(self.checkpoint_path)  
        self._build_dataloader()
        print("finish dataloading")
        self._build_model()
        print("finish building model")
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        print("starting build the dataset")
        
        if args.graph_type=='line_graph':
          dataset = SemiEvalDataset(root_dir=args.root_dir,
                              topology_builder=LineBasedGraphConstruction,
                              graph_type='static',
                              pretrained_word_emb_file=args.pre_word_emb_file,
                              topology_subdir='LineGraph')
        elif args.graph_type=='dependency_graph':
          dataset = SemiEvalDataset(root_dir=args.root_dir,
                              topology_builder=DependencyBasedGraphConstruction,
                              graph_type='static',
                              pretrained_word_emb_file=args.pre_word_emb_file,
                              topology_subdir='DependencyGraph') 
        elif args.graph_type=='node_emb':
          dataset = SemiEvalDataset(root_dir=args.root_dir,
                              topology_builder=NodeEmbeddingBasedGraphConstruction,
                              graph_type='dynamic',
                              pretrained_word_emb_file=args.pre_word_emb_file,
                              topology_subdir='DynamicGraph_node_emb',
                              merge_strategy=None,
                              dynamic_graph_type=args.graph_type if args.graph_type in ('node_emb', 'node_emb_refined') else None)
        elif args.graph_type=='node_emb_refined':
            if args.init_graph_type == 'line':
                dynamic_init_topology_builder = LineBasedGraphConstruction
            elif args.init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction_without_tokenizer
            elif args.init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            elif args.init_graph_type == 'ie':
                merge_strategy = 'global'
                dynamic_init_topology_builder = IEBasedGraphConstruction
            else:
                # init_topology_builder
                raise RuntimeError('Define your own init_topology_builder')  
            dataset = SemiEvalDataset(root_dir=args.root_dir,
                              topology_builder=NodeEmbeddingBasedRefinedGraphConstruction,
                              graph_type='dynamic',
                              pretrained_word_emb_file=args.pre_word_emb_file,
                              topology_subdir='DynamicGraph_node_emb_refined',
                              tag_types=self.tag_types,
                              dynamic_graph_type=args.graph_type if args.graph_type in ('node_emb', 'node_emb_refined') else None,
                              dynamic_init_topology_builder=dynamic_init_topology_builder,
                              dynamic_init_topology_aux_args={'dummy_param': 0})          
          
        print("strating loading the training data")
        self.train_dataloader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True,
                                           collate_fn=dataset.collate_fn)

        print("strating loading the testing data")
        self.test_dataloader = DataLoader(dataset.test, batch_size=args.batch_size, shuffle=False,
                                          collate_fn=dataset.collate_fn)
        print("strating loading the vocab")
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Word2tag(self.vocab,device=self.device).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters,lr=args.lr, weight_decay=args.weight_decay)
    
    def _build_evaluation(self):
        self.metrics = Accuracy('F1')

    def train(self):
        max_score = -1
        max_idx=0
        for epoch in range(args.epochs):
            # score_test = self.evaluate_test(epoch)
            self.model.train()
            print("Epoch: {}".format(epoch))
            pred_collect = []
            gt_collect = []
            
            for data in self.train_dataloader:
                graph_list, tgt, pos_info = data
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pos_info_l= [pos for pos in pos_info]
                pred_tags, loss = self.model(graph_list, tgt_l, pos_info_l, require_loss=True)
                pred_collect.extend(pred_tags)   #pred: list of batch_sentence pred tensor         
                gt_collect.extend(tgt)  #tgt:list of sentence token tensor                
                #num_tokens=len(torch.cat(pred_tags).view(-1))
                print("Epoch: {}".format(epoch)+" loss:"+str(loss.cpu().item()))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step() 
                
            if epoch%1 ==0:
                # score = self.evaluate_train(epoch)
                score=self.evaluate_test(epoch)
                if score>max_score:
                    torch.save(self.model.state_dict(),self.checkpoint_path+'checkpoint_best')
                    max_idx=epoch
                max_score = max(max_score, score) 
                                              
        return max_score,max_idx

    def evaluate_train(self,epoch):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        with torch.no_grad():
            for data in self.train_dataloader:
                graph_list, tgt, pos_info = data
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pos_info_l= [pos for pos in pos_info]
                pred, loss = self.model(graph_list[40:], tgt_l[40:], pos_info_l[40:], require_loss=True)
                pred_collect.extend(pred)   #pred: list of batch_sentence pred tensor         
                gt_collect.extend(tgt[40:])  #tgt:list of sentence token tensor  
                
        f1 = f1_score(torch.tensor(gt_collect).cpu().numpy(),  torch.tensor(pred_collect), average='macro')
        print("Testing results: f1 is %5.2f"%(f1*100))         
        print("Epoch: {}".format(epoch)+" loss:"+str(loss.cpu().item())) 
                                                  
        return f1 
    
    def evaluate_test(self, epoch):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        with torch.no_grad():
            for data in self.test_dataloader:
                graph_list, tgt, pos_info = data
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pos_info_l= [pos for pos in pos_info]
                pred, loss = self.model(graph_list, tgt_l, pos_info_l, require_loss=True)
                pred_collect.extend(pred)   #pred: list of batch_sentence pred tensor         
                gt_collect.extend(tgt)  #tgt:list of sentence token tensor  
                
        f1 = f1_score(torch.tensor(gt_collect).view(-1).cpu().numpy(),  torch.tensor(pred_collect).view(-1), average='macro')
        print("Testing results: f1 is %5.2f"%(f1*100))         
        # print("Epoch: {}".format(epoch)+" loss:"+str(loss.cpu().item()))
                                                  
        return f1     
    
    def test(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path+'checkpoint_best'))
        print("sucessfully loaded the existing saved model!")
        self.model.eval()
        pred_collect = []
        gt_collect=[]
        with torch.no_grad():
            for data in self.test_dataloader:
                graph_list, tgt, pos_info = data
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pos_info_l= [pos.to(self.device) for pos in pos_info]
                pred, loss = self.model(graph_list, tgt_l, pos_info_l, require_loss=True)
                pred_collect.extend(pred)   #pred: list of batch_sentence pred tensor         
                gt_collect.extend(tgt)  #tgt:list of sentence token tensor  
                
        f1 = f1_score(torch.tensor(gt_collect).view(-1).cpu().numpy(),  torch.tensor(pred_collect).view(-1), average='macro')
        print("Testing results: f1 is %5.2f"%(f1*100))         
        return f1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='relation extraction')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--direction_option", type=str, default='undirected',
                        help="direction type (`undirected`, `bi_fuse`, `bi_sep`)")
    parser.add_argument("--root_dir", type=str, default='/raid/ghn/graph4nlp/examples/pytorch/relation_extraction/dataset',
                        help="")
    parser.add_argument('--mlp_layers', type=int, default=1, help='Number of output mlp layers.')
    parser.add_argument("--lstm_num_layers", type=int, default=1,
                        help="number of hidden layers in lstm")
    parser.add_argument("--gnn_num_layers", type=int, default=1,
                        help="number of hidden layers in gnn")    
    parser.add_argument("--init_hidden_size", type=int, default=300,
                        help="initial_emb_hidden_size")
    parser.add_argument("--hidden_size", type=int, default=200,
                        help="initial_emb_hidden_size")      
    parser.add_argument("--num_class", type=int, default=19,
                        help="num_class")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--word_dropout", type=float, default=0.1,
                        help="input feature dropout")
    parser.add_argument("--gnn_dropout", type=float, default=0.2,
                        help="input feature dropout")    
    parser.add_argument("--rnn_dropout", type=list, default=0.2,
                        help="dropout for rnn in word_emb")      
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help="weight decay")
    parser.add_argument('--aggregate_type', type=str, default='mean',
                        help="aggregate type: 'mean','gcn','pool','lstm'")
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        help="ingnn type: 'gat','graphsage','ggnn'")  
    parser.add_argument('--use_gnn', type=bool, default=True,
                        help="whether to use gnn")      
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size for training")
    parser.add_argument("--graph_type", type=str, default="dependency_graph",
                        help="graph_type:line_graph, dependency_graph, dynamic_graph")    
    parser.add_argument("--init_graph_type", type=str, default="dependency",
                        help="initial graph construction type ('line', 'dependency', 'constituency', 'ie')")      
    parser.add_argument("--pre_word_emb_file", type=str, default=  "F:/xiaojie/G2GIE/embeddings/glove.840B.300d.txt",
                        help="path of pretrained_word_emb_file")     
    parser.add_argument("--gl_num_heads", type=int, default=1,
                        help="num of heads for dynamic graph construction")  
    parser.add_argument("--gl_epsilon", type=int, default=0.5,
                        help="epsilon for graph sparsification")  
    parser.add_argument("--gl_top_k", type=int, default= None,
                        help="top k for graph sparsification")      
    parser.add_argument("--gl_smoothness_ratio", type=float, default= None,
                        help="smoothness ratio for graph regularization loss")
    parser.add_argument("--gl_sparsity_ratio", type=float, default= None,
                        help="sparsity ratio for graph regularization loss")
    parser.add_argument("--gl_connectivity_ratio", type=float, default= None,
                        help="connectivity ratio for graph regularization loss")    
    parser.add_argument("--init_adj_alpha", type=float, default=0.8,
                        help="alpha ratio for combining initial graph adjacency matrix") 
    parser.add_argument("--gl_metric_type", type=str, default='weighted_cosine',
                        help="similarity metric type for dynamic graph construction ('weighted_cosine', 'attention', 'rbf_kernel', 'cosine')" )   
    parser.add_argument("--no_fix_word_emb", type=bool, default=True,
                        help="Not fix pretrained word embeddings (default: false)" )   
    parser.add_argument("--no_fix_bert_emb", type=bool, default=False,
                        help="Not fix pretrained word embeddings (default: false)" )   
  

    args = parser.parse_args() 
    runner = SemiEval()
    max_score, max_idx=runner.train()
    print("Train finish, best score: {:.3f}".format(max_score))
    print(max_idx)
    score = runner.test()

