import copy
import random
import time
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.mawps import MawpsDatasetForTree
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.graph_construction import *

warnings.filterwarnings('ignore')


class Mawps:
    def __init__(self, opt=None):
        super(Mawps, self).__init__()
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
        self.data_dir = self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"]
        self._build_model()
        self._build_dataloader()

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
            raise NotImplementedError

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
                    'pretrained_word_emb_cache_dir': self.opt["pretrained_word_emb_cache_dir"],
                    "for_inference": 1,
                    'reused_vocab_model': self.model.vocab_model
                    }

        # for inference
        inference_dataset = MawpsDatasetForTree(**para_dic)

        self.inference_data_loader = DataLoader(inference_dataset.test, batch_size=1, shuffle=False, num_workers=0,
                                                collate_fn=inference_dataset.collate_fn)

        self.vocab_model = self.model.vocab_model
        self.src_vocab = self.vocab_model.in_word_vocab
        self.tgt_vocab = self.vocab_model.out_word_vocab
        self.share_vocab = self.vocab_model.share_vocab if self.use_share_vocab else None

    def _build_model(self):
        '''For encoder-decoder'''
        self.model = Graph2Tree.load_checkpoint(self.opt["checkpoint_save_path"], "best.pt").to(self.device)

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

    def infer(self):
        self.model.eval()
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
                reference = self.model.tgt_vocab.get_symbol_idx_for_list(batch_original_tree_list[0].split())
                eval_vocab = self.tgt_vocab

            candidate = self.model.translate(eval_input_graph,
                                        oov_dict=oov_dict,
                                        use_beam_search=True,
                                        beam_size=self.opt["beam_size"])

            candidate = [int(c) for c in candidate]
            print(" ".join(x['token'] for x in eval_input_graph.node_attributes))
            print(" ".join(self.model.tgt_vocab.get_idx_symbol_for_list(candidate)) + '\n')


if __name__ == "__main__":
    from config import get_args

    start = time.time()
    runner = Mawps(opt=get_args())
    runner.infer()

    end = time.time()
    print("total time: {} minutes\n".format((end - start) / 60))
