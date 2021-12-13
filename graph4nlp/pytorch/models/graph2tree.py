import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from graph4nlp.pytorch.models.base import Graph2XBase
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import DecoderStrategy
from graph4nlp.pytorch.modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder
from graph4nlp.pytorch.modules.utils.tree_utils import Tree, to_cuda


class Graph2Tree(Graph2XBase):
    """
        Graph2Tree is a general end-to-end neural encoder-decoder model that maps an input graph
        to a tree structure. The graph2tree model consists the following components: 1) node
        embedding 2) graph embedding 3) tree decoding. Since the full pipeline will consist all
        parameters, so we will add prefix to the original parameters in each component as
        follows (except the listed four parameters):
            1) emb_ + parameter_name (eg: ``emb_input_size``)
            2) gnn_ + parameter_name (eg: ``gnn_direction_option``)
            3) dec_ + parameter_name (eg: ``dec_max_decoder_step``)
        Considering neatness, we will only present the four hyper-parameters which don't meet
        regulations.
    Parameters
    ----------
    vocab_model: VocabModel
        The vocabulary.
    graph_name: str
        The graph type. Excepted in ["dependency", "constituency", "node_emb", "node_emb_refined"].
    gnn: str
        The graph neural network type. Expected in ["gcn", "gat", "graphsage", "ggnn"]
    embedding_style: dict
        The options used in the embedding module.
    """

    def __init__(
        self,
        vocab_model,
        embedding_style,
        graph_name,
        # embedding
        emb_input_size,
        emb_hidden_size,
        emb_word_dropout,
        emb_rnn_dropout,
        emb_fix_word_emb,
        emb_fix_bert_emb,
        # gnn
        gnn,
        gnn_num_layers,
        gnn_direction_option,
        gnn_input_size,
        gnn_hidden_size,
        gnn_output_size,
        gnn_feat_drop,
        gnn_attn_drop,
        # decoder
        dec_use_copy,
        dec_hidden_size,
        dec_dropout,
        dec_teacher_forcing_rate,
        dec_max_decoder_step,
        dec_max_tree_depth,
        dec_attention_type,
        dec_use_sibling,
        # optional
        criterion=None,
        share_vocab=False,
        **kwargs
    ):
        super(Graph2Tree, self).__init__(
            vocab_model=vocab_model,
            emb_input_size=emb_input_size,
            emb_hidden_size=emb_hidden_size,
            graph_name=graph_name,
            gnn_direction_option=gnn_direction_option,
            gnn=gnn,
            gnn_num_layers=gnn_num_layers,
            embedding_style=embedding_style,
            gnn_feats_dropout=gnn_feat_drop,
            gnn_attn_dropout=gnn_attn_drop,
            emb_rnn_dropout=emb_rnn_dropout,
            emb_fix_word_emb=emb_fix_word_emb,
            emb_fix_bert_emb=emb_fix_bert_emb,
            emb_word_dropout=emb_word_dropout,
            gnn_hidden_size=gnn_hidden_size,
            gnn_input_size=gnn_input_size,
            gnn_output_size=gnn_output_size,
            **kwargs
        )

        self.src_vocab, self.tgt_vocab = vocab_model.in_word_vocab, vocab_model.out_word_vocab
        self.gnn_hidden_size = gnn_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.use_copy = dec_use_copy
        self.input_size = self.src_vocab.vocab_size
        self.output_size = self.tgt_vocab.vocab_size
        self.criterion = (
            nn.NLLLoss(
                size_average=False,
                ignore_index=self.src_vocab.get_symbol_idx(self.src_vocab.pad_token),
            )
            if criterion is None
            else criterion
        )
        self.use_share_vocab = share_vocab
        if self.use_share_vocab == 0:
            self.tgt_word_embedding = nn.Embedding(
                self.tgt_vocab.vocab_size,
                dec_hidden_size,
                padding_idx=self.tgt_vocab.get_symbol_idx(self.tgt_vocab.pad_token),
                _weight=torch.from_numpy(self.tgt_vocab.embeddings).float(),
            )

        self.decoder = StdTreeDecoder(
            attn_type=dec_attention_type,
            embeddings=self.enc_word_emb.word_emb_layer
            if self.use_share_vocab
            else self.tgt_word_embedding,
            enc_hidden_size=gnn_hidden_size,
            dec_emb_size=self.tgt_vocab.embedding_dims,
            dec_hidden_size=dec_hidden_size,
            output_size=self.output_size,
            criterion=self.criterion,
            teacher_force_ratio=dec_teacher_forcing_rate,
            use_sibling=dec_use_sibling,
            use_copy=self.use_copy,
            dropout_for_decoder=dec_dropout,
            max_dec_seq_length=dec_max_decoder_step,
            max_dec_tree_depth=dec_max_tree_depth,
            tgt_vocab=self.tgt_vocab,
        )

    def forward(self, batch_graph, tgt_tree_batch, oov_dict=None):
        batch_graph = self.graph_initializer(batch_graph)
        if hasattr(self, "graph_topology") and hasattr(self.graph_topology, "dynamic_topology"):
            batch_graph = self.graph_topology.dynamic_topology(batch_graph)
        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features["node_feat"]

        loss = self.decoder(g=batch_graph, tgt_tree_batch=tgt_tree_batch, oov_dict=oov_dict)
        return loss

    def translate(self, input_graph, use_beam_search=True, beam_size=4, oov_dict=None):
        device = input_graph.device
        prev_c = torch.zeros((1, self.dec_hidden_size), requires_grad=False)
        prev_h = torch.zeros((1, self.dec_hidden_size), requires_grad=False)

        batch_graph = self.graph_initializer(input_graph)
        if hasattr(self, "graph_topology") and hasattr(self.graph_topology, "dynamic_topology"):
            batch_graph = self.graph_topology.dynamic_topology(batch_graph)

        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features["rnn_emb"] = batch_graph.node_features["node_feat"]

        params = self.decoder._extract_params(batch_graph)
        graph_node_embedding = params["graph_node_embedding"]
        if self.decoder.graph_pooling_strategy == "max":
            graph_level_embedding = torch.max(graph_node_embedding, 1)[0]
        rnn_node_embedding = params["rnn_node_embedding"]
        # graph_node_mask = params["graph_node_mask"]
        enc_w_list = params["enc_batch"]

        enc_outputs = graph_node_embedding
        prev_c = graph_level_embedding
        prev_h = graph_level_embedding

        # decode
        queue_decode = []
        queue_decode.append({"s": (prev_c, prev_h), "parent": 0, "child_index": 1, "t": Tree()})
        head = 1
        while head <= len(queue_decode) and head <= self.decoder.max_dec_tree_depth:
            s = queue_decode[head - 1]["s"]
            parent_h = s[1]
            t = queue_decode[head - 1]["t"]

            # sibling_state = torch.zeros(
            #     (1, self.dec_hidden_size), dtype=torch.float, requires_grad=False
            # ).to(device)

            # flag_sibling = False
            # for q_index in range(len(queue_decode)):
            #     if (
            #         (head <= len(queue_decode))
            #         and (q_index < head - 1)
            #         and (queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"])
            #         and (
            #         queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]
            #         )
            #     ):
            #         flag_sibling = True
            #         sibling_index = q_index
            # if flag_sibling:
            #     sibling_state = queue_decode[sibling_index]["s"][1]

            if head == 1:
                prev_word = torch.tensor(
                    [self.tgt_vocab.get_symbol_idx(self.tgt_vocab.start_token)], dtype=torch.long
                )
            else:
                prev_word = torch.tensor([self.tgt_vocab.get_symbol_idx("(")], dtype=torch.long)

            prev_word = to_cuda(prev_word, device)
            i_child = 1
            if not use_beam_search:
                while True:
                    prediction, (curr_c, curr_h), _ = self.decoder.decode_step(
                        tgt_batch_size=1,
                        dec_single_input=prev_word,
                        dec_single_state=s,
                        memory=enc_outputs,
                        parent_state=parent_h,
                        oov_dict=oov_dict,
                        enc_batch=enc_w_list,
                    )
                    s = (curr_c, curr_h)
                    prev_word = torch.log(prediction + 1e-31)
                    prev_word = prev_word.argmax(1)

                    if (
                        int(prev_word[0]) == self.tgt_vocab.get_symbol_idx(self.tgt_vocab.end_token)
                        or t.num_children >= self.decoder.max_dec_seq_length
                    ):
                        break
                    elif int(prev_word[0]) == self.tgt_vocab.get_symbol_idx(
                        self.tgt_vocab.non_terminal_token
                    ):
                        queue_decode.append(
                            {
                                "s": (s[0].clone(), s[1].clone()),
                                "parent": head,
                                "child_index": i_child,
                                "t": Tree(),
                            }
                        )
                        t.add_child(int(prev_word[0]))
                    else:
                        t.add_child(int(prev_word[0]))
                    i_child = i_child + 1
            else:
                topk = 1
                # decoding goes sentence by sentence
                assert graph_node_embedding.size(0) == 1
                beam_search_generator = DecoderStrategy(
                    beam_size=beam_size,
                    vocab=self.tgt_vocab,
                    decoder=self.decoder,
                    rnn_type="lstm",
                    use_copy=True,
                    use_coverage=False,
                )
                decoded_results = beam_search_generator.beam_search_for_tree_decoding(
                    decoder_initial_state=(s[0], s[1]),
                    decoder_initial_input=prev_word,
                    parent_state=parent_h,
                    graph_node_embedding=enc_outputs,
                    rnn_node_embedding=rnn_node_embedding,
                    device=device,
                    topk=topk,
                    oov_dict=oov_dict,
                    enc_batch=enc_w_list,
                )
                generated_sentence = decoded_results[0][0]
                for node_i in generated_sentence:
                    if int(node_i.wordid.item()) == self.tgt_vocab.get_symbol_idx(
                        self.tgt_vocab.non_terminal_token
                    ):
                        queue_decode.append(
                            {
                                "s": (node_i.h[0].clone(), node_i.h[1].clone()),
                                "parent": head,
                                "child_index": i_child,
                                "t": Tree(),
                            }
                        )
                        t.add_child(int(node_i.wordid.item()))
                        i_child = i_child + 1
                    elif (
                        int(node_i.wordid.item())
                        != self.tgt_vocab.get_symbol_idx(self.tgt_vocab.end_token)
                        and int(node_i.wordid.item())
                        != self.tgt_vocab.get_symbol_idx(self.tgt_vocab.start_token)
                        and int(node_i.wordid.item()) != self.tgt_vocab.get_symbol_idx("(")
                    ):
                        t.add_child(int(node_i.wordid.item()))
                        i_child = i_child + 1

            head = head + 1
        for i in range(len(queue_decode) - 1, 0, -1):
            cur = queue_decode[i]
            queue_decode[cur["parent"] - 1]["t"].children[cur["child_index"] - 1] = cur["t"]
        return queue_decode[0]["t"].to_list(self.tgt_vocab)

    def init(self, init_weight):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if (
                    ("word_embedding" in name)
                    or ("word_emb_layer" in name)
                    or ("bert_embedding" in name)
                ):
                    pass
                else:
                    if len(param.size()) >= 2:
                        if "rnn" in name:
                            init.orthogonal_(param)
                        else:
                            init.xavier_uniform_(param, gain=1.0)
                    else:
                        init.uniform_(param, -init_weight, init_weight)

    def post_process(self, decode_results, vocab):
        candidate = [int(c) for c in decode_results]
        pred_str = " ".join(self.tgt_vocab.get_idx_symbol_for_list(candidate))
        return [pred_str]

    def inference_forward(self, batch_graph, beam_size, topk=1, oov_dict=None):
        """
            Decoding with the support of beam_search.
            Specifically, when ``beam_size`` is 1, it is equal to greedy search.
        Parameters
        ----------
        batch_graph: GraphData
            The graph input
        beam_size: int
            The beam width. When it is 1, the output is equal to greedy search's output.
        topk: int, default=1
            The number of decoded output to be reserved.
            Usually, ``topk`` should be smaller or equal to ``beam_size``
        oov_dict: VocabModel, default=None
            The vocabulary for copy.

        Returns
        -------
        results: torch.Tensor
            The results with the shape of ``[batch_size, topk, max_decoder_step]`` containing the word indexes. # noqa
        """
        return self.translate(
            input_graph=batch_graph["graph_data"],
            use_beam_search=(beam_size > 1),
            beam_size=beam_size,
            oov_dict=oov_dict,
        )

    @classmethod
    def from_args(cls, opt, vocab_model):
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
        initializer_args = cls._get_node_initializer_params(opt)
        gnn_args = cls._get_gnn_params(opt)
        dec_args = cls._get_decoder_params(opt)

        args = copy.deepcopy(initializer_args)
        args.update(gnn_args)
        args.update(dec_args)
        args["share_vocab"] = opt["graph_construction_args"]["graph_construction_share"][
            "share_vocab"
        ]
        return cls(vocab_model=vocab_model, **args)

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
            private_args["activation"] = (
                getattr(F, private_args["activation"]) if private_args["activation"] else None
            )
        if "norm" in private_args.keys():
            private_args["norm"] = (
                getattr(F, private_args["norm"]) if private_args["norm"] else None
            )

        gnn_shared_args = {"gnn_" + key: value for key, value in shared_args.items()}
        pri_shared_args = {"gnn_" + key: value for key, value in private_args.items()}
        ret = copy.deepcopy(dict(gnn_shared_args, **pri_shared_args))
        ret["gnn"] = opt["graph_embedding_name"]
        return ret

    @staticmethod
    def _get_node_initializer_params(opt):
        # Dynamic graph construction related params are stored here
        init_args = opt["graph_construction_args"]["graph_construction_private"]
        ret: dict = copy.deepcopy(init_args)
        args = opt["graph_initialization_args"]
        ret.update(args)
        ret.pop("embedding_style")
        emb_ret = {"emb_" + key: value for key, value in ret.items()}
        emb_ret["embedding_style"] = args["embedding_style"]
        emb_ret["graph_name"] = opt["graph_construction_name"]
        return emb_ret
