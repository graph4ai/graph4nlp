import copy
import random
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph4nlp.pytorch.datasets.jobs import JobsDatasetForTree
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.utils.tree_utils import Tree

warnings.filterwarnings('ignore')


class Jobs:
    def __init__(self, opt=None):
        super(Jobs, self).__init__()
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
        self.make_inference = self.opt["make_inference"]==1
        self.data_dir = self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"]
        self.inference_data_dir = self.opt["graph_construction_args"]["graph_construction_share"]["inference_root_dir"] if self.make_inference else None

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        graph_type = self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"]
        enc_emb_size = self.opt["graph_construction_args"]["node_embedding"]["input_size"]
        tgt_emb_size = self.opt["decoder_args"]["rnn_decoder_share"]["input_size"]
        topology_subdir = self.opt["graph_construction_args"]["graph_construction_share"]["topology_subdir"]
        dynamic_init_topology_builder = None

        if graph_type == "dependency":
            my_topology_builder = DependencyBasedGraphConstruction
            my_graph_type = 'static'
        elif graph_type == "constituency":
            my_topology_builder = DependencyBasedGraphConstruction
            my_graph_type = 'static'
        elif graph_type == "node_emb":
            my_topology_builder = NodeEmbeddingBasedGraphConstruction
            my_graph_type = 'dynamic'
        elif graph_type == "node_emb_refined":
            my_topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            my_graph_type = 'dynamic'

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
        else:
            raise NotImplementedError
        
        para_dic =  {'root_dir': self.data_dir,
                    'word_emb_size': enc_emb_size,
                    'topology_builder': my_topology_builder,
                    'topology_subdir': topology_subdir, 
                    'edge_strategy': self.opt["graph_construction_args"]["graph_construction_private"]["edge_strategy"],
                    'graph_type': my_graph_type,
                    'dynamic_graph_type': graph_type, 
                    'share_vocab': self.use_share_vocab, 
                    'enc_emb_size': enc_emb_size,
                    'dec_emb_size': tgt_emb_size,
                    'dynamic_init_topology_builder': dynamic_init_topology_builder,
                    'min_word_vocab_freq': self.opt["min_freq"],
                    'pretrained_word_emb_name': self.opt["pretrained_word_emb_name"],
                    'pretrained_word_emb_url': self.opt["pretrained_word_emb_url"], 
                    'pretrained_word_emb_cache_dir': self.opt["pretrained_word_emb_cache_dir"]
                    }

        dataset = JobsDatasetForTree(**para_dic)

        # for inference
        para_dic['root_dir'] = self.inference_data_dir
        para_dic['for_inference'] = self.make_inference
        para_dic['train_root'] = self.data_dir
        inference_dataset = JobsDatasetForTree(**para_dic)

        para_dic = {'root_dir': self.data_dir,
                    'word_emb_size': enc_emb_size,
                    'topology_builder': my_topology_builder,
                    'topology_subdir': topology_subdir,
                    'edge_strategy': self.opt["graph_construction_args"]["graph_construction_private"]["edge_strategy"],
                    'graph_type': my_graph_type,
                    'dynamic_graph_type': graph_type,
                    'share_vocab': self.use_share_vocab,
                    'enc_emb_size': enc_emb_size,
                    'dec_emb_size': tgt_emb_size,
                    'dynamic_init_topology_builder': dynamic_init_topology_builder,
                    'min_word_vocab_freq': self.opt["min_freq"],
                    'pretrained_word_emb_name': self.opt["pretrained_word_emb_name"],
                    'pretrained_word_emb_url': self.opt["pretrained_word_emb_url"],
                    'pretrained_word_emb_cache_dir': self.opt["pretrained_word_emb_cache_dir"]
                    }

        dataset = JobsDatasetForTree(**para_dic)

        self.train_data_loader = DataLoader(dataset.train, batch_size=self.opt["batch_size"], shuffle=True,
                                            num_workers=0,
                                            collate_fn=dataset.collate_fn)
        self.test_data_loader = DataLoader(dataset.test, batch_size=1, shuffle=False, num_workers=1,
                                           collate_fn=dataset.collate_fn)
        self.inference_data_loader = DataLoader(inference_dataset.test, batch_size=1, shuffle=False, num_workers=1,
                                          collate_fn=inference_dataset.collate_fn)
        self.vocab_model = dataset.vocab_model
        self.src_vocab = self.vocab_model.in_word_vocab
        self.tgt_vocab = self.vocab_model.out_word_vocab
        self.share_vocab = self.vocab_model.share_vocab if self.use_share_vocab else None
        # if self.use_share_vocab:
        #     self.share_vocab = dataset.share_vocab_model
        # self.vocab_model = VocabForAll(in_word_vocab=self.src_vocab, out_word_vocab=self.tgt_vocab, share_vocab=self.share_vocab)

    def _build_model(self):
        '''For encoder-decoder'''
        self.model = Graph2Tree.from_args(self.opt, vocab_model=self.vocab_model)
        self.model.init(self.opt["init_weight"])
        self.model.to(self.device)

    def _build_optimizer(self):
        optim_state = {"learningRate": self.opt["learning_rate"], "weight_decay": self.opt["weight_decay"]}
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=optim_state['learningRate'],
                                    weight_decay=optim_state['weight_decay'])

    def prepare_ext_vocab(self, batch_graph, src_vocab):
        oov_dict = copy.deepcopy(src_vocab)
        token_matrix = []
        for n in batch_graph.node_attributes:
            node_token = n['token']
            if (n.get('type') == None or n.get('type') == 0) and oov_dict.get_symbol_idx(
                    node_token) == oov_dict.get_symbol_idx(oov_dict.unk_token):
                oov_dict.add_symbol(node_token)
            token_matrix.append(oov_dict.get_symbol_idx(node_token))
        batch_graph.node_features['token_id_oov'] = torch.tensor(token_matrix, dtype=torch.long).to(self.device)
        return oov_dict

    def train_epoch(self, epoch):
        loss_to_print = 0
        num_batch = len(self.train_data_loader)
        for step, data in tqdm(enumerate(self.train_data_loader), desc=f'Epoch {epoch:02d}',
                               total=len(self.train_data_loader)):
            batch_graph, batch_tree_list, batch_original_tree_list = data['graph_data'], data['dec_tree_batch'], data[
                'original_dec_tree_batch']
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
            loss = self.model(batch_graph, batch_tree_list_refined if self.use_copy else batch_tree_list,
                              oov_dict=oov_dict)
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.opt["grad_clip"])
            self.optimizer.step()
            loss_to_print += loss
        print("-------------\nLoss = {:.3f}".format(loss_to_print / num_batch))

    def train(self):
        print("-------------\nStarting training.")
        best_acc = 0.0
        for epoch in range(1, self.opt["max_epochs"] + 1):
            self.model.train()
            self.train_epoch(epoch)
            if epoch >= 5:
                val_acc = self.eval((self.model))
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.model.save_checkpoint(self.opt["checkpoint_save_path"], "best.pt")
                    print("Best Model Saved!")
        print(f"Best Accuracy: {val_acc:.4f}")
        print("="*20 + "making inference" + "="*20)
        self.infer(self.model) # replace the model with the best model you saved.

    def eval(self, model):
        from evaluation import convert_to_string, compute_tree_accuracy
        model.eval()
        reference_list = []
        candidate_list = []
        for data in tqdm(self.test_data_loader, desc="Eval: "):
            eval_input_graph, batch_tree_list, batch_original_tree_list = data['graph_data'], data['dec_tree_batch'], \
                                                                          data['original_dec_tree_batch']
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

            candidate = model.translate(eval_input_graph,
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
        eval_acc = compute_tree_accuracy(
            candidate_list, reference_list, eval_vocab)
        print(f"Accuracy: {eval_acc:.4f}\n")
        return eval_acc


    def infer(self, model):
        model.eval()
        for data in self.inference_data_loader:
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

            candidate = model.translate(eval_input_graph,
                                        oov_dict=oov_dict,
                                        use_beam_search=True,
                                        beam_size=self.opt["beam_size"])

            candidate = [int(c) for c in candidate]
            print(" ".join(x['token'] for x in eval_input_graph.node_attributes))
            print(" ".join(model.tgt_vocab.get_idx_symbol_for_list(candidate)) + '\n')

if __name__ == "__main__":
    from config import get_args

    start = time.time()
    runner = Jobs(opt=get_args())

    runner.train()

    end = time.time()
    print("total time: {} minutes\n".format((end - start) / 60))
