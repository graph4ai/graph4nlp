import copy

import torch.nn.functional as F
import torch

from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from .base import Graph2XBase
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import DecoderStrategy
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from graph4nlp.pytorch.data.data import GraphData


class Graph2Seq(Graph2XBase):
    """
        The graph2seq model consists the following components: 1) node embedding 2) graph embedding 3) decoding.
        Since the full pipeline will consist all parameters, so we will add prefix to the original parameters
         in each component as follows (except the listed four parameters):
            1) emb_ + parameter_name (eg: ``emb_input_size``)
            2) gnn_ + parameter_name (eg: ``gnn_direction_option``)
            3) dec_ + parameter_name (eg: ``dec_max_decoder_step``)
        Considering neatness, we will only present the four hyper-parameters which don't meet regulations.
        Examples
        -------
        # Build a vocab model from scratch
        >>> "It is just a how-to-use example."
        >>> from graph4nlp.pytorch.modules.config import get_basic_args
        >>> opt = get_basic_args(graph_construction_name="node_emb", graph_embedding_name="gat", decoder_name="stdrnn")
        >>> graph2seq = Graph2Seq.from_args(opt=opt, vocab_model=vocab_model, device=torch.device("cuda:0"))
        >>> graph_list = [GraphData() for _ in range(2)]
        >>> tgt_seq = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> seq_out, _, _ = graph2seq(graph_list=graph_list, tgt_seq=tgt_seq)
        >>> print(seq_out.shape) # [2, 6, 5] (assume the vocabulary size is 5 and max_decoder_step is 6)
    Parameters
    ----------
    vocab_model: VocabModel
        The vocabulary.
    graph_type: str
        The graph type. Excepted in ["dependency", "constituency", "node_emb", "node_emb_refined"].
    gnn: str
        The graph neural network's name. Expected in ["gcn", "gat", "graphsage", "ggnn"]
    device: torch.device
        The device.
    """
    def __init__(self, vocab_model, emb_input_size, emb_hidden_size, embedding_style,
                 graph_type, gnn_direction_option, gnn_input_size, gnn_hidden_size, gnn_output_size,
                 gnn, gnn_num_layers, dec_hidden_size,
                 # dropout
                 gnn_feat_drop=0.0, gnn_attn_drop=0.0,
                 emb_fix_word_emb=False, emb_fix_bert_emb=False, emb_word_dropout=0.0, emb_rnn_dropout=0.0,
                 dec_max_decoder_step=50,
                 dec_use_copy=False, dec_use_coverage=False,
                 dec_graph_pooling_strategy=None, dec_rnn_type="lstm", dec_tgt_emb_as_output_layer=False,
                 dec_attention_type="uniform", dec_fuse_strategy="average", dec_node_type_num=None,
                 dec_dropout=0.0, device=None,
                 **kwargs):
        super(Graph2Seq, self).__init__(vocab_model=vocab_model, emb_input_size=emb_input_size, emb_hidden_size=emb_hidden_size,
                                        graph_type=graph_type, gnn_direction_option=gnn_direction_option,
                                        gnn=gnn, gnn_num_layers=gnn_num_layers, embedding_style=embedding_style,
                                        device=device, gnn_feats_dropout=gnn_feat_drop,
                                        gnn_attn_dropout=gnn_attn_drop,
                                        emb_rnn_dropout=emb_rnn_dropout, emb_fix_word_emb=emb_fix_word_emb,
                                        emb_fix_bert_emb=emb_fix_bert_emb,
                                        emb_word_dropout=emb_word_dropout,
                                        gnn_hidden_size=gnn_hidden_size, gnn_input_size=gnn_input_size,
                                        gnn_output_size=gnn_output_size,
                                        **kwargs)

        self.use_copy = dec_use_copy
        self.use_coverage = dec_use_coverage
        self.dec_rnn_type = dec_rnn_type

        self._build_decoder(rnn_type=dec_rnn_type, decoder_length=dec_max_decoder_step, vocab_model=vocab_model,
                            word_emb=self.word_emb, rnn_input_size=emb_hidden_size,
                            input_size=2 * gnn_hidden_size if gnn_direction_option == 'bi_sep' else gnn_hidden_size,
                            hidden_size=dec_hidden_size, graph_pooling_strategy=dec_graph_pooling_strategy,
                            use_copy=dec_use_copy, use_coverage=dec_use_coverage,
                            tgt_emb_as_output_layer=dec_tgt_emb_as_output_layer,
                            attention_type=dec_attention_type, node_type_num=dec_node_type_num,
                            fuse_strategy=dec_fuse_strategy,
                            rnn_dropout=dec_dropout)

    def _build_decoder(self, decoder_length, input_size, rnn_input_size, hidden_size, graph_pooling_strategy,
                       vocab_model, word_emb,
                       use_copy=False, use_coverage=False, tgt_emb_as_output_layer=False,
                       rnn_type="lstm", attention_type="uniform", node_type_num=None, fuse_strategy="average",
                       rnn_dropout=0.2):

        self.seq_decoder = StdRNNDecoder(rnn_type=rnn_type, max_decoder_step=decoder_length,
                                         input_size=input_size,
                                         hidden_size=hidden_size, graph_pooling_strategy=graph_pooling_strategy,
                                         word_emb=word_emb, vocab=vocab_model.out_word_vocab,
                                         attention_type=attention_type, fuse_strategy=fuse_strategy,
                                         node_type_num=node_type_num,
                                         rnn_emb_input_size=rnn_input_size, use_coverage=use_coverage,
                                         use_copy=use_copy,
                                         tgt_emb_as_output_layer=tgt_emb_as_output_layer, dropout=rnn_dropout)

    def encoder_decoder(self, batch_graph, old_graph_list, oov_dict=None, tgt_seq=None):
        # run GNN
        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']
        graph_list_decoder = from_batch(batch_graph)
        if self.use_copy and "token_id_oov" not in batch_graph.node_features.keys():
            for g, g_ori in zip(graph_list_decoder, old_graph_list):
                g.node_features['token_id_oov'] = g_ori.node_features['token_id_oov']

        # down-task
        prob, enc_attn_weights, coverage_vectors = self.seq_decoder(graph_list_decoder, tgt_seq=tgt_seq,
                                                                    oov_dict=oov_dict)
        return prob, enc_attn_weights, coverage_vectors

    def encoder_decoder_beam_search(self, batch_graph, old_graph_list, beam_size, topk=1, oov_dict=None):
        generator = DecoderStrategy(beam_size=beam_size, vocab=self.seq_decoder.vocab, rnn_type=self.dec_rnn_type,
                                    decoder=self.seq_decoder, use_copy=self.use_copy,
                                    use_coverage=self.use_coverage)
        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']
        graph_list_decoder = from_batch(batch_graph)
        if self.use_copy and "token_id_oov" not in batch_graph.node_features.keys():
            for g, g_ori in zip(graph_list_decoder, old_graph_list):
                g.node_features['token_id_oov'] = g_ori.node_features['token_id_oov']
        beam_results = generator.generate(graph_list=graph_list_decoder, oov_dict=oov_dict, topk=topk)
        return beam_results

    def forward(self, graph_list, tgt_seq=None, oov_dict=None):
        batch_graph = self.graph_topology(graph_list)
        return self.encoder_decoder(batch_graph=batch_graph, old_graph_list=graph_list,
                                    oov_dict=oov_dict, tgt_seq=tgt_seq)

    def translate(self, graph_list, beam_size, topk=1, oov_dict=None):
        """
            Decoding with the support of beam_search.
            Specifically, when ``beam_size`` is 1, it is equal to greedy search.
        Parameters
        ----------
        graph_list: list[GraphData]
            The graph input
        beam_size: int
            The beam width. When it is 1, the output is equal to greedy search's output.
        topk: int, default=1
            The number of decoded sequence to be reserved.
            Usually, ``topk`` should be smaller or equal to ``beam_size``
        oov_dict: VocabModel, default=None
            The vocabulary for copy.

        Returns
        -------
        results: torch.Tensor
            The results with the shape of ``[batch_size, topk, max_decoder_step]`` containing the word indexes.
        """

        batch_graph = self.graph_topology(graph_list)
        return self.encoder_decoder_beam_search(batch_graph=batch_graph, old_graph_list=graph_list, beam_size=beam_size,
                                                topk=topk, oov_dict=oov_dict)

    @classmethod
    def from_args(cls, opt, vocab_model, device):
        """
            The function for building ``Graph2Seq`` model.
        Parameters
        ----------
        opt: dict
            The configuration dict. It should has the same hierarchy and keys as the template.
        vocab_model: VocabModel
            The vocabulary.
        device: torch.device
            The device.

        Returns
        -------
        model: Graph2Seq
        """
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
