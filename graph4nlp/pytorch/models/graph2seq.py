from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from .base import Graph2XBase


class Graph2Seq(Graph2XBase):
    def __init__(self, vocab_model, encoder_hidden_size, decoder_hidden_size,
                 graph_type, direction_option,
                 gnn, gnn_layer_number, embedding_style,
                 device, max_decoder_depth=50,
                 use_copy=False, use_coverage=False,
                 graph_pooling_strategy=None, rnn_type="lstm", tgt_emb_as_output_layer=False, attention_type="uniform",
                 fuse_strategy="average",
                 node_type_num=None, feats_dropout=0.2, attn_dropout=0.2, rnn_dropout=0.2, fix_word_emb=False,
                 **kwargs):
        super(Graph2Seq, self).__init__(vocab_model=vocab_model, encoder_hidden_size=encoder_hidden_size,
                                        graph_type=graph_type, direction_option=direction_option,
                                        gnn=gnn, gnn_layer_number=gnn_layer_number, embedding_style=embedding_style,
                                        device=device, feats_dropout=feats_dropout, attn_dropout=attn_dropout,
                                        rnn_dropout=rnn_dropout, fix_word_emb=fix_word_emb, **kwargs)

        self.use_copy = use_copy
        self.use_coverage = use_coverage

        self._build_decoder(rnn_type=rnn_type, decoder_length=max_decoder_depth, vocab_model=vocab_model,
                            word_emb=self.word_emb,
                            input_size=2 * encoder_hidden_size if direction_option == 'bi_sep' else encoder_hidden_size,
                            hidden_size=decoder_hidden_size, graph_pooling_strategy=graph_pooling_strategy,
                            use_copy=use_copy, use_coverage=use_coverage,
                            tgt_emb_as_output_layer=tgt_emb_as_output_layer,
                            attention_type=attention_type, node_type_num=node_type_num, fuse_strategy=fuse_strategy,
                            rnn_dropout=rnn_dropout)

    def _build_decoder(self, decoder_length, input_size, hidden_size, graph_pooling_strategy, vocab_model, word_emb,
                       use_copy=False, use_coverage=False, tgt_emb_as_output_layer=False,
                       rnn_type="lstm", attention_type="uniform", node_type_num=None, fuse_strategy="average",
                       rnn_dropout=0.2):

        self.seq_decoder = StdRNNDecoder(rnn_type=rnn_type, max_decoder_step=decoder_length,
                                         input_size=input_size,
                                         hidden_size=hidden_size, graph_pooling_strategy=graph_pooling_strategy,
                                         word_emb=word_emb, vocab=vocab_model.out_word_vocab,
                                         attention_type=attention_type, fuse_strategy=fuse_strategy,
                                         node_type_num=node_type_num,
                                         rnn_emb_input_size=hidden_size, use_coverage=use_coverage, use_copy=use_copy,
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
        print(opt)
        exit(0)
        return cls(vocab_model=vocab_model, device=device)
