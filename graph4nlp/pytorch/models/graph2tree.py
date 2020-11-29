import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from graph4nlp.pytorch.data.data import from_batch

from .base import Graph2XBase
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import DecoderStrategy
from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder, create_mask
from graph4nlp.pytorch.modules.utils.tree_utils import DataLoaderForGraphEncoder, Tree, Vocab, to_cuda

from graph4nlp.pytorch.data.data import GraphData


class Graph2Tree(Graph2XBase):
    def __init__(self, vocab_model,
                 emb_input_size, 
                 emb_hidden_size, 
                 embedding_style,
                 graph_type, 
                 gnn_direction_option, 
                 gnn_input_size, 
                 gnn_hidden_size, 
                 gnn_output_size,
                 gnn, 
                 gnn_num_layers, 
                 dec_hidden_size,
                 gnn_feat_drop=0.0, 
                 gnn_attn_drop=0.0,
                 emb_fix_word_emb=False, 
                 emb_fix_bert_emb=False, 
                 emb_word_dropout=0.0, 
                 emb_rnn_dropout=0.0,
                 dec_max_decoder_step=50,
                 dec_max_tree_depth=50, 
                 dec_use_copy=False, 
                 dec_use_sibling=False,
                 dec_graph_pooling_strategy="max", 
                 dec_rnn_type="lstm", 
                 dec_attention_type="uniform", 
                 dec_dropout=0.0, 
                 dec_teacher_force_ratio=1.0, 
                 device=None, 
                 **kwargs):
        super(Graph2Tree, self).__init__(vocab_model=vocab_model, 
                                         emb_input_size=emb_input_size,
                                         emb_hidden_size=emb_hidden_size,
                                         graph_type=graph_type, 
                                         gnn_direction_option=gnn_direction_option,
                                         gnn=gnn,
                                         gnn_num_layers=gnn_num_layers,
                                         embedding_style=embedding_style,
                                         device=device,
                                         gnn_feats_dropout=gnn_feat_drop,
                                         gnn_attn_dropout=gnn_attn_drop,
                                         emb_rnn_dropout=emb_rnn_dropout,
                                         emb_fix_word_emb=emb_fix_word_emb,
                                         emb_fix_bert_emb=emb_fix_bert_emb,
                                         emb_word_dropout=emb_word_dropout,
                                         gnn_hidden_size=gnn_hidden_size,
                                         gnn_input_size=gnn_input_size,
                                         gnn_output_size=gnn_output_size,
                                         **kwargs)

        self.use_copy = dec_use_copy
        self.device = device

        self._build_decoder(decoder_length=dec_max_decoder_step,
                            decoder_tree_depth=dec_max_tree_depth,
                            vocab_model=vocab_model,
                            word_emb=self.word_emb,
                            rnn_input_size=emb_hidden_size,
                            input_size=2 * gnn_hidden_size if gnn_direction_option == 'bi_sep' else gnn_hidden_size,
                            hidden_size=dec_hidden_size,
                            use_copy=dec_use_copy,
                            rnn_type=dec_rnn_type,
                            attention_type=dec_attention_type,
                            dec_dropout=dec_dropout,
                            device=device,
                            teacher_force_ratio=dec_teacher_force_ratio,
                            use_sibling=dec_use_sibling,
                            graph_pooling_strategy=dec_graph_pooling_strategy)

    def _build_decoder(self, decoder_length,
                       decoder_tree_depth,
                       vocab_model,
                       word_emb,
                       rnn_input_size,
                       input_size,
                       hidden_size,
                       use_copy=False,
                       rnn_type="lstm",
                       attention_type="uniform",
                       dec_dropout=0.1,
                       device=None,
                       teacher_force_ratio=1.0,
                       use_sibling=False,
                       graph_pooling_strategy="max"):
        
        self.decoder = StdTreeDecoder(attn_type=attention_type, 
                                      embeddings=word_emb,
                                      enc_hidden_size=hidden_size,
                                      dec_emb_size=vocab_model.out_word_vocab.embedding_dims,
                                      dec_hidden_size=hidden_size,
                                      output_size=vocab_model.out_word_vocab.vocab_size,
                                      device=device,
                                      criterion=nn.NLLLoss(size_average=False),
                                      teacher_force_ratio=teacher_force_ratio,
                                      use_sibling=False,
                                      use_attention=True, 
                                      use_copy=use_copy,
                                      use_coverage=False,
                                      fuse_strategy="average",
                                      num_layers=1,
                                      dropout_for_decoder=dec_dropout,
                                      rnn_type=rnn_type,
                                      max_dec_seq_length=decoder_length,
                                      max_dec_tree_depth=decoder_tree_depth,
                                      tgt_vocab=vocab_model.out_word_vocab,
                                      graph_pooling_strategy=graph_pooling_strategy)

    def forward(self, graph_list, tgt_tree_batch, oov_dict=None):
        batch_graph = self.graph_topology(graph_list)
        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

        batch_graph_list_decoder_input = from_batch(batch_graph)
        if self.use_copy and "token_id_oov" not in batch_graph.node_features.keys():
            for g, g_ in zip(batch_graph_list_decoder_input, graph_list):
                g.node_features['token_id_oov'] = g_.node_features['token_id_oov']

        loss = self.decoder(g=batch_graph_list_decoder_input,
                            tgt_tree_batch=tgt_tree_batch, oov_dict=oov_dict)
        return loss

    def init(self, init_weight):
        to_cuda(self.gnn_encoder, self.device)
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

    @classmethod
    def from_args(cls, opt, vocab_model, device):
        emb_args = cls._get_node_embedding_params(opt)
        gnn_args = cls._get_gnn_params(opt)
        dec_args = cls._get_decoder_params(opt)

        args = (copy.deepcopy(emb_args))
        args.update(gnn_args)
        args.update(dec_args)

        return cls(vocab_model=vocab_model, device=device, **args)

    @staticmethod
    def _get_decoder_params(opt):
        dec_args = opt["decoder_args"]
        shared_args = copy.deepcopy(dec_args["rnn_decoder_share"])
        private_args = copy.deepcopy(dec_args["rnn_decoder_private"])
        ret = copy.deepcopy(dict(shared_args, **private_args))
        dec_ret = {"dec_" + key: value for key, value in ret.items()}
        return dec_ret

    @staticmethod
    def _get_gnn_params(opt):
        args = opt["graph_embedding_args"]
        shared_args = copy.deepcopy(args["graph_embedding_share"])
        private_args = copy.deepcopy(args["graph_embedding_private"])
        if "activation" in private_args.keys():
            private_args["activation"] = getattr(F, private_args["activation"]) if private_args["activation"] else None
        if "norm" in private_args.keys():
            private_args["norm"] = getattr(F, private_args["norm"]) if private_args["norm"] else None

        gnn_shared_args = {"gnn_" + key: value for key, value in shared_args.items()}
        pri_shared_args = {"gnn_" + key: value for key, value in private_args.items()}
        ret = copy.deepcopy(dict(gnn_shared_args, **pri_shared_args))
        ret["gnn"] = opt["graph_embedding_name"]
        return ret

    @staticmethod
    def _get_node_embedding_params(opt):
        args = opt["graph_construction_args"]["node_embedding"]
        ret: dict = copy.deepcopy(args)
        ret.pop("embedding_style")
        emb_ret = {"emb_" + key: value for key, value in ret.items()}
        emb_ret["embedding_style"] = args["embedding_style"]
        emb_ret["graph_type"] = opt["graph_construction_name"]
        return emb_ret
