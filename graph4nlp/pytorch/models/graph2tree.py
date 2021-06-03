import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from graph4nlp.pytorch.models.base import Graph2XBase
from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder


class Graph2Tree(Graph2XBase):
    def __init__(self, vocab_model, device, embedding_style, graph_type, 
                 # embedding
                 emb_input_size, emb_hidden_size, emb_word_dropout, emb_rnn_dropout, emb_fix_word_emb, emb_fix_bert_emb, 
                 # gnn
                 gnn, gnn_num_layers, gnn_direction_option, gnn_input_size, gnn_hidden_size, gnn_output_size, gnn_feat_drop, gnn_attn_drop, 
                 # decoder
                 dec_use_copy, dec_hidden_size, dec_dropout, dec_teacher_forcing_rate, dec_max_decoder_step, dec_max_tree_depth, dec_attention_type, dec_use_sibling,
                 # optional
                 criterion=None,
                 share_vocab=False,
                 **kwargs):
        super(Graph2Tree, self).__init__(vocab_model=vocab_model, emb_input_size=emb_input_size, emb_hidden_size=emb_hidden_size,
                                        graph_type=graph_type, gnn_direction_option=gnn_direction_option,
                                        gnn=gnn, gnn_num_layers=gnn_num_layers, embedding_style=embedding_style,
                                        gnn_feats_dropout=gnn_feat_drop,
                                        gnn_attn_dropout=gnn_attn_drop,
                                        emb_rnn_dropout=emb_rnn_dropout, emb_fix_word_emb=emb_fix_word_emb,
                                        emb_fix_bert_emb=emb_fix_bert_emb,
                                        emb_word_dropout=emb_word_dropout,
                                        gnn_hidden_size=gnn_hidden_size, gnn_input_size=gnn_input_size,
                                        gnn_output_size=gnn_output_size,
                                        **kwargs)

        self.src_vocab, self.tgt_vocab = vocab_model.in_word_vocab, vocab_model.out_word_vocab
        self.device = device
        self.use_copy = dec_use_copy
        self.input_size = self.src_vocab.vocab_size
        self.output_size = self.tgt_vocab.vocab_size
        self.criterion = nn.NLLLoss(size_average=False) if criterion == None else criterion
        self.use_share_vocab = share_vocab
        if self.use_share_vocab == 0:
            self.tgt_word_embedding = nn.Embedding(self.tgt_vocab.vocab_size, dec_hidden_size, 
                                                padding_idx=self.tgt_vocab.get_symbol_idx(self.tgt_vocab.pad_token),
                                                _weight=torch.from_numpy(self.tgt_vocab.embeddings).float())

        self.decoder = StdTreeDecoder(attn_type=dec_attention_type,
                                      embeddings=self.enc_word_emb.word_emb_layer if self.use_share_vocab else self.tgt_word_embedding,
                                      enc_hidden_size=gnn_hidden_size,
                                      dec_emb_size=self.tgt_vocab.embedding_dims,
                                      dec_hidden_size=dec_hidden_size,
                                      output_size=self.output_size,
                                      device=self.device,
                                      criterion=self.criterion,
                                      teacher_force_ratio=dec_teacher_forcing_rate,
                                      use_sibling=dec_use_sibling,
                                      use_copy=self.use_copy,
                                      dropout_for_decoder=dec_dropout,
                                      max_dec_seq_length=dec_max_decoder_step,
                                      max_dec_tree_depth=dec_max_tree_depth,
                                      tgt_vocab=self.tgt_vocab)

    def forward(self, batch_graph, tgt_tree_batch, oov_dict=None):
        batch_graph = self.graph_topology(batch_graph)
        batch_graph = self.encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

        loss = self.decoder(g=batch_graph, tgt_tree_batch=tgt_tree_batch, oov_dict=oov_dict)
        return loss

    def init(self, init_weight):
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        for name, param in self.named_parameters():
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
    def from_args(cls, opt, vocab_model, device=None):
        """
            The function for building ``Graph2Tree`` model.
        Parameters
        ----------
        opt: dict
            The configuration dict. It should has the same hierarchy and keys as the template.
        vocab_model: VocabModel
            The vocabulary.

        Returns
        -------
        model: Graph2Tree
        """
        emb_args = cls._get_node_embedding_params(opt)
        gnn_args = cls._get_gnn_params(opt)
        dec_args = cls._get_decoder_params(opt)

        args = (copy.deepcopy(emb_args))
        args.update(gnn_args)
        args.update(dec_args)
        args["share_vocab"] = opt["graph_construction_args"]["graph_construction_share"]["share_vocab"]

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
