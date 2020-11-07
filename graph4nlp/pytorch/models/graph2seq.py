from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from .base import Graph2XBase

import copy
import torch.nn.functional as F

class Graph2Seq(Graph2XBase):
    def __init__(self, vocab_model, emb_hidden_size, embedding_style,
                 graph_type, gnn_direction_option, gnn_hidden_size,
                 gnn, gnn_num_layers, dec_hidden_size,
                 # dropout
                 emb_word_dropout=0.0, emb_rnn_dropout=0.0,

                 gnn_feats_dropout=0.0, gnn_attn_dropout=0.0,
                 device=None, emb_fix_word_emb=False,
                 emb_fix_bert_emb=False,
                 dec_max_decoder_step=50,
                 dec_use_copy=False, dec_use_coverage=False,
                 dec_graph_pooling_strategy=None, dec_rnn_type="lstm", dec_tgt_emb_as_output_layer=False, dec_attention_type="uniform",
                 dec_fuse_strategy="average",
                 dec_node_type_num=None, dec_dropout=0.0,
                 **kwargs):
        super(Graph2Seq, self).__init__(vocab_model=vocab_model, emb_hidden_size=emb_hidden_size,
                                        graph_type=graph_type, gnn_direction_option=gnn_direction_option,
                                        gnn=gnn, gnn_layer_number=gnn_num_layers, embedding_style=embedding_style,
                                        device=device, gnn_feats_dropout=gnn_feats_dropout, gnn_attn_dropout=gnn_attn_dropout,
                                        emb_rnn_dropout=emb_rnn_dropout, emb_fix_word_emb=emb_fix_word_emb, emb_fix_bert_emb=emb_fix_bert_emb,
                                        emb_word_dropout=emb_word_dropout, gnn_hidden_size=gnn_hidden_size, **kwargs)

        self.use_copy = dec_use_copy
        self.use_coverage = dec_use_coverage

        self._build_decoder(rnn_type=dec_rnn_type, decoder_length=dec_max_decoder_step, vocab_model=vocab_model,
                            word_emb=self.word_emb, rnn_input_size=emb_hidden_size,
                            input_size=2 * gnn_hidden_size if gnn_direction_option == 'bi_sep' else gnn_hidden_size,
                            hidden_size=dec_hidden_size, graph_pooling_strategy=dec_graph_pooling_strategy,
                            use_copy=dec_use_copy, use_coverage=dec_use_coverage,
                            tgt_emb_as_output_layer=dec_tgt_emb_as_output_layer,
                            attention_type=dec_attention_type, node_type_num=dec_node_type_num, fuse_strategy=dec_fuse_strategy,
                            rnn_dropout=dec_dropout)

    def _build_decoder(self, decoder_length, input_size, rnn_input_size, hidden_size, graph_pooling_strategy, vocab_model, word_emb,
                       use_copy=False, use_coverage=False, tgt_emb_as_output_layer=False,
                       rnn_type="lstm", attention_type="uniform", node_type_num=None, fuse_strategy="average",
                       rnn_dropout=0.2):

        self.seq_decoder = StdRNNDecoder(rnn_type=rnn_type, max_decoder_step=decoder_length,
                                         input_size=input_size,
                                         hidden_size=hidden_size, graph_pooling_strategy=graph_pooling_strategy,
                                         word_emb=word_emb, vocab=vocab_model.out_word_vocab,
                                         attention_type=attention_type, fuse_strategy=fuse_strategy,
                                         node_type_num=node_type_num,
                                         rnn_emb_input_size=rnn_input_size, use_coverage=use_coverage, use_copy=use_copy,
                                         tgt_emb_as_output_layer=tgt_emb_as_output_layer, dropout=rnn_dropout)

    def forward(self, graph_list, tgt_seq=None, oov_dict=None):
        batch_graph = self.graph_topology(graph_list)
        # run GNN
        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']
        graph_list_decoder = from_batch(batch_graph)
        if self.use_copy and "token_id_oov" not in batch_graph.node_features.keys():
            for g, g_ori in zip(graph_list_decoder, graph_list):
                g.node_features['token_id_oov'] = g_ori.node_features['token_id_oov']

        # down-task
        prob, enc_attn_weights, coverage_vectors = self.seq_decoder(graph_list_decoder, tgt_seq=tgt_seq,
                                                                    oov_dict=oov_dict)
        return prob, enc_attn_weights, coverage_vectors

    @classmethod
    def from_args(cls, opt, vocab_model, device):
        emb_args = cls._get_node_embedding_params(opt)
        gnn_args = cls._get_gnn_params(opt)
        dec_args = cls._get_decoder_params(opt)

        args = (copy.deepcopy(emb_args))
        args.update(gnn_args)
        args.update(dec_args)
        print(args)

        return cls(vocab_model=vocab_model, device=device, **args)

    @staticmethod
    def _get_decoder_params(opt):
        dec_args = opt.decoder_args
        shared_args = copy.deepcopy(dec_args["rnn_decoder_share"])
        private_args = copy.deepcopy(dec_args["rnn_decoder_private"])
        ret = copy.deepcopy(dict(shared_args, **private_args))
        dec_ret = {"dec_" + key: value for key, value in ret.items()}
        return dec_ret

    @staticmethod
    def _get_gnn_params(opt):
        args = opt.graph_embedding_args
        shared_args = copy.deepcopy(args["graph_embedding_share"])
        private_args = copy.deepcopy(args["graph_embedding_private"])
        if "activation" in private_args.keys():
            private_args["activation"] = getattr(F, private_args["activation"]) if private_args["activation"] else None
        if "norm" in private_args.keys():
            private_args["norm"] = getattr(F, private_args["norm"]) if private_args["norm"] else None

        gnn_shared_args = {"gnn_" + key: value for key, value in shared_args.items()}
        pri_shared_args = {"gnn_" + key: value for key, value in private_args.items()}
        ret = copy.deepcopy(dict(gnn_shared_args, **pri_shared_args))
        ret["gnn"] = opt.graph_embedding_name
        return ret

    @staticmethod
    def _get_node_embedding_params(opt):
        args = opt.graph_constrcution_args["node_embedding"]
        ret: dict = copy.deepcopy(args)
        # ret.pop("hidden_size")
        # ret.pop("word_dropout")
        # ret.pop("rnn_dropout")
        # ret["emb_word_dropout"] = args["word_dropout"]
        # ret["emb_rnn_dropout"] = args["rnn_dropout"]
        ret.pop("embedding_style")
        emb_ret = {"emb_" + key: value for key, value in ret.items()}
        emb_ret["embedding_style"] = args["embedding_style"]
        emb_ret["graph_type"] = opt.graph_construction_name
        return emb_ret

