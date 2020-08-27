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
from efficiency.log import show_time, show_var, fwrite
from graph4nlp.pytorch.data.data import *
from graph4nlp.pytorch.datasets.conll import ConllDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction_without_tokenize import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.line_graph_construction import LineBasedGraphConstruction
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda

from graph4nlp.pytorch.modules.graph_embedding.graphsage import GraphSAGE
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import dgl
from torchcrf import CRF

from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.prediction.classification.node_classification.FeedForwardNN import FeedForwardNN 
from graph4nlp.pytorch.modules.prediction.classification.node_classification.BiLSTMFeedForwardNN import BiLSTMFeedForwardNN 
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from conlleval import evaluate


def conll_score(preds, tgts,tag_types):
    #preds is a list and each elements is the list of tags of a sentence
    #tgts is a lits and each elements is the tensor of tags of a text
    pred_list=[]
    tgt_list=[]
    
    for idx in range(len(preds)):
        pred_list.append(preds[idx].cpu().clone().numpy())
    for idx in range(len(tgts)):
        tgt_list.extend(tgts[idx].cpu().clone().numpy().tolist())    
    pred_tags=[tag_types[int(pred)] for pred in pred_list]
    tgt_tags=[tag_types[int(tgt)] for tgt in tgt_list]   
    prec, rec, f1 = evaluate(tgt_tags, pred_tags, verbose=False) 
    return prec, rec, f1


def logits2tag(logits):
        pred=F.softmax(logits).argmax(-1)
        return pred  

def write_file(tokens_collect,pred_collect,tag_collect,file_name,tag_types):
    num_sent=len(tokens_collect)
    f=open(file_name,'w')
    for idx in range(num_sent):
            sent_token=tokens_collect[idx]
            sent_pred=pred_collect[idx].cpu().clone().numpy()
            sent_tag=tag_collect[idx].cpu().clone().numpy()
            #f.write('%s\n' % ('-X- SENTENCE START'))             
            for word_idx in range(len(sent_token)):             
                w=sent_token[word_idx]
                tgt=tag_types[sent_tag[word_idx].item()]
                pred=tag_types[sent_pred[word_idx].item()]
                f.write('%d %s %s %s\n' % (word_idx + 1, w, tgt, pred))
    
    f.close()
    
def get_tokens(g_list):
    tokens=[]
    for g in g_list:
        sent_token=[]
        dic=g.node_attributes
        for node in dic:
            sent_token.append(dic[node]['token'])
            if 'ROOT' in sent_token:
               sent_token.remove('ROOT')
        tokens.append(sent_token)    
    return tokens   

class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super(CRFLayer, self).__init__()  
        self.model=CRF(num_tags, batch_first=True)
        
    def forward(self,batch_graph,tgt_tags): 
        #prepare emission_score
        sent_graph_list=from_batch(batch_graph)
        emb_list=[]
        #sent_len=[]
        for g in sent_graph_list:
            emb_list.append(g.node_features['logits'])
            #sent_len.append(g.node_features['logits'].size()[0]) 
        emission_padded=pad_sequence(emb_list,batch_first=True)
        
        #mask=torch.zeros(emission_padded.size(0),emission_padded.size(1))
        #for i in range(len(sent_len)):
        #    mask[i,:sent_len[i]]=1
            
        #prepare tags
        tag_padded=pad_sequence(tgt_tags,batch_first=True,padding_value=-1)#batch*seq_len
        
        #prepare the mask
        mask=(tag_padded>-1)
            
        loss=-self.model(emission_padded, tag_padded,mask)
        pred_tags=self.model.decode(emission_padded,mask)
        pred_tags_=[torch.tensor(pred) for pred in pred_tags]
        
        return loss,pred_tags_
      
        
        

class SentenceBiLSTMCRF(nn.Module):
    def __init__(self,input_size,output_size,hidden_size=None,device=None,use_crf=False):
        super(SentenceBiLSTMCRF, self).__init__()
        self.prediction=BiLSTMFeedForwardNN(input_size,output_size,hidden_size).to(device)
        self.crf=CRFLayer(8).to(device)
        self.general_loss=GeneralLoss("CrossEntropy")
        self.use_crf=use_crf
    def forward(self,batch_graph,tgt_tags):
        #text_graph to sentence_graph
#        sent_graph_list=[]
#        for g in text_graph_list:
#            nodes=g.node_attributes
#            sent_g={}
#            for node_idx,node_attribute in nodes.items():
#               node_feature=g.get_node_features(node_idx)
#               sentence_id=node_attribute['sentence_id']
#               if sentence_id in sent_g.keys():               
#                  sent_g[sentence_id].add_nodes(1)
#                  node_id=len(sent_g[sentence_id].node_attributes)
#                  sent_g[sentence_id].node_attributes[node_id-1]=node_attribute
#                  sent_g[sentence_id].node_features['node_emb'][node_id-1]=node_feature
#               else:
#                  sent_g[sentence_id]=GraphData()
#                  sent_g[sentence_id].add_nodes(1) 
#                  sent_g[sentence_id].node_attributes[0]=node_attribute 
#                  sent_g[sentence_id].node_features['node_emb'][node_id-1]=node_feature                  
#            sent_graph_list.extend(list(sent_g.values()))  #each graph is a  sentence
            
        #go through the bilstm function:
        #batch_graph=GraphData.to_batch(sent_graph_list)
        batch_graph= self.prediction(batch_graph)
         
        if self.use_crf is False:
            tgt=torch.cat(tgt_tags)
            logits=batch_graph.node_features['logits'][:,:] #[batch*sentence*num_nodes,num_lable]
            loss=self.general_loss(logits,tgt)
            pred_tags=logits2tag(logits)
        
        else:
           #go through CRF
           loss,pred_tags=self.crf(batch_graph,tgt_tags)
           pred_tags=torch.cat(pred_tags).view(-1)
         
        return loss, pred_tags
         
                   
class Word2tag(nn.Module):
    def __init__(self, vocab, device=None):
        super(Word2tag, self).__init__()
        self.vocab = vocab
        self.device =device
        
        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"}
        if args.graph_type=='line_graph':
          self.graph_topology = LineBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=args.init_hidden_size, dropout=args.dropout, use_cuda=(self.device != None),
                                                               fix_word_emb=False)
        if args.graph_type=='dependency_graph':
          self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=args.init_hidden_size, dropout=args.dropout, use_cuda=(self.device != None),
                                                               fix_word_emb=False)          
        if args.graph_type=='dynamic_graph':
             raise NotImplementedError('dynamic graph: Not Implemented.')    
             
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer.to(self.device)
        
        self.gnn_type=args.gnn_type
        self.use_gnn=args.use_gnn
        use_crf=args.use_crf
        direction_option=args.direction_option
        
        if self.use_gnn is False:            
           self.bilstmcrf=SentenceBiLSTMCRF(args.init_hidden_size,args.num_class, args.lstm_hidden_size, device=self.device, use_crf=use_crf).to(self.device) 
        else:   
            if self.gnn_type=="graphsage":            
               self.gnn = GraphSAGE(args.gnn_num_layers, args.init_hidden_size, args.init_hidden_size, args.init_hidden_size, aggregator_type='mean',direction_option=direction_option, activation=F.elu).to(self.device)        
               self.bilstmcrf=SentenceBiLSTMCRF(args.init_hidden_size, args.num_class,args.lstm_hidden_size, device=self.device, use_crf=args.use_crf).to(self.device)
            elif self.gnn_type=="ggnn":
               self.gnn = GraphSAGE(args.gnn_num_layers, args.init_hidden_size, args.init_hidden_size,direction_option=direction_option).to(self.device)        
               self.bilstmcrf=SentenceBiLSTMCRF(args.init_hidden_size, args.num_class,args.lstm_hidden_size, device=self.device, use_crf=use_crf).to(self.device)            
            elif self.gnn_type=="gat":
               heads = 0
               self.gnn = GAT(args.gnn_num_layers,args.init_hidden_size,args.init_hidden_size,args.init_hidden_size, heads,direction_option=direction_option,feat_drop=0.6, attn_drop=0.6, negative_slope=0.2, activation=F.elu).to(self.device)        
               self.bilstmcrf=SentenceBiLSTMCRF(args.init_hidden_size, args.num_class,args.lstm_hidden_size, device=self.device, use_crf=use_crf).to(self.device)
            
     

    def forward(self, graph_list, tgt=None, require_loss=True):
        batch_graph = self.graph_topology(graph_list)

        if self.use_gnn is False:
            batch_graph.node_features['node_emb']=batch_graph.node_features['node_feat']
        else:
           # run GNN
           batch_graph = self.gnn(batch_graph)

        
        # down-task
        loss,pred=self.bilstmcrf(batch_graph,tgt)
        self.loss=loss
        
        if require_loss==True:
           return pred, self.loss
        else:
            loss=None
            return pred, self.loss
        

class Conll:
    def __init__(self):
        super(Conll, self).__init__()
        self.tag_types=['I-PER', 'O', 'B-ORG', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'B-MISC']
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
          dataset = ConllDataset(root_dir="graph4nlp/pytorch/test/dataset/conll",
                              topology_builder=LineBasedGraphConstruction,
                              topology_subdir='LineGraph',
                              tag_types=self.tag_types)
        elif args.graph_type=='dependency_graph':
          dataset = ConllDataset(root_dir="graph4nlp/pytorch/test/dataset/conll",
                              topology_builder=LineBasedGraphConstruction,
                              topology_subdir='DependencyGraph',
                              tag_types=self.tag_types) 
        elif args.graph_type=='dynamic_graph':
          dataset = ConllDataset(root_dir="graph4nlp/pytorch/test/dataset/conll",
                              topology_builder=LineBasedGraphConstruction,
                              topology_subdir='DynamicGraph',
                              tag_types=self.tag_types)           
          
        print("strating loading the training data")
        self.train_dataloader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True,
                                           num_workers=1,
                                           collate_fn=dataset.collate_fn)
        print("strating loading the validating data")
        self.val_dataloader = DataLoader(dataset.val, batch_size=100, shuffle=True,
                                          num_workers=1,
                                          collate_fn=dataset.collate_fn)
        print("strating loading the testing data")
        self.test_dataloader = DataLoader(dataset.test, batch_size=100, shuffle=True,
                                          num_workers=1,
                                          collate_fn=dataset.collate_fn)        
        print("strating loading the vocab")
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Word2tag(self.vocab,device=self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters,lr=args.lr, weight_decay=args.weight_decay)
    
    def _build_evaluation(self):
        self.metrics = Accuracy(['F1','precision','recall'])

    def train(self):
        max_score = -1
        for epoch in range(args.epochs):
            self.model.train()
            print("Epoch: {}".format(epoch))
            pred_collect = []
            gt_collect = []
            for data in self.train_dataloader:
                graph_list, tgt = data
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pred_tags, loss = self.model(graph_list, tgt_l, require_loss=True)
                pred_collect.extend(pred_tags)   #pred: list of batch_sentence pred tensor         
                gt_collect.extend(tgt)  #tgt:list of sentence token tensor                
                #num_tokens=len(torch.cat(pred_tags).view(-1))
                print("Epoch: {}".format(epoch)+" loss:"+str(loss.cpu().item()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
            #pred_tensor=torch.cat(pred_collect).reshape(-1).cpu()
            #gt_tensor=torch.cat(gt_collect).view(-1).cpu()               
            #train_score = self.metrics.calculate_scores(ground_truth=gt_tensor, predict=pred_tensor,average='macro')[2]              
            #print("epoch:"+str(epoch)+", training F1 accuracy: {:.3f}".format(train_score))        
            if epoch%1 ==0:
                score = self.evaluate(epoch)
                if score>max_score:
                    torch.save(self.model.state_dict(),self.checkpoint_path+'checkpoint_best')
                max_score = max(max_score, score)                
        return max_score

    def evaluate(self,epoch):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        tokens_collect=[]
        with torch.no_grad():
            for data in self.val_dataloader:
                graph_list, tgt = data
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pred,loss= self.model(graph_list, tgt_l, require_loss=True)
                pred_collect.extend(pred)   #pred: list of batch_sentence pred tensor         
                gt_collect.extend(tgt)  #tgt:list of sentence token tensor
                tokens_collect.extend(get_tokens(graph_list))  
                
        prec, rec, f1 = conll_score(pred_collect,gt_collect,self.tag_types)
        print("Testing results: precision is %5.2f, rec is %5.2f, f1 is %5.2f"%(prec,rec,f1))          
        print("Epoch: {}".format(epoch)+" loss:"+str(loss.cpu().item())) 
        
#        pred_tensor=torch.cat(pred_collect).reshape(-1).cpu()
#        #num_tokens=len(pred_tensor)
#        gt_tensor=torch.cat(gt_collect).view(-1).cpu()               
#        score = self.metrics.calculate_scores(ground_truth=gt_tensor, predict=pred_tensor,average='macro')[2]              
#        print("epoch:"+str(epoch)+", validation F1 accuracy: {:.3f}".format(score))    
                                           
        return f1 
    
    def test(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path+'checkpoint_best'))
        print("sucessfully loaded the existing saved model!")
        self.model.eval()
        pred_collect = []
        tokens_collect = []
        tgt_collect=[]
        with torch.no_grad():       
            for data in self.test_dataloader:
                graph_list, tgt = data
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pred,loss = self.model(graph_list, tgt_l,require_loss=True)
                #pred = logits2tag(g)
                pred_collect.extend(pred)
                tgt_collect.extend(tgt)             
                tokens_collect.extend(get_tokens(graph_list))  
        prec, rec, f1 = conll_score(pred_collect,tgt_collect,self.tag_types)
        print("Testing results: precision is %5.2f, rec is %5.2f, f1 is %5.2f"%(prec,rec,f1))          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--direction_option", type=str, default='bi_sep',
                        help="direction type (`uni`, `bi_fuse`, `bi_sep`)")
    parser.add_argument("--lstm_num_layers", type=int, default=1,
                        help="number of hidden layers in lstm")
    parser.add_argument("--gnn_num_layers", type=int, default=1,
                        help="number of hidden layers in gnn")    
    parser.add_argument("--init_hidden_size", type=int, default=300,
                        help="initial_emb_hidden_size")
    parser.add_argument("--lstm_hidden_size", type=int, default=50,
                        help="initial_emb_hidden_size")    
    parser.add_argument("--num_class", type=int, default=8,
                        help="hiddensize")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help="weight decay")
    parser.add_argument('--aggregate_type', type=str, default='mean',
                        help="aggregate type: 'mean','gcn','pool','lstm'")
    parser.add_argument('--use_crf', type=bool, default=False,
                        help="indicates whether to use crf as last layer")
    parser.add_argument('--gnn_type', type=str, default='graphsage',
                        help="ingnn type: 'gat','graphsage','ggnn'")  
    parser.add_argument('--use_gnn', type=bool, default=False,
                        help="whether to use gnn")      
    parser.add_argument("--batch_size", type=int, default=150,
                        help="batch size for training")
    parser.add_argument("--graph_type", type=str, default="line_graph",
                        help="graph_type:line_graph, dependency_graph, dynamic_graph")    
    
    args = parser.parse_args()    
    # preprocess()
    runner = Conll()
    #max_score=runner.train()
    #print("Train finish, best score: {:.3f}".format(max_score))
    runner.test()
    



