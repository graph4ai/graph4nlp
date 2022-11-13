from graph4nlp.pytorch.models.graph2seq import Graph2Seq

from examples.pytorch.rgcn.rgcn import RGCN


class RGCNGraph2Seq(Graph2Seq):
    def __init__(
        self,
        vocab_model,
        emb_input_size,
        emb_hidden_size,
        embedding_style,
        graph_construction_name,
        gnn_direction_option,
        gnn_input_size,
        gnn_hidden_size,
        gnn_output_size,
        gnn,
        gnn_num_layers,
        dec_hidden_size,
        share_vocab=False,
        gnn_feat_drop=0,
        gnn_attn_drop=0,
        emb_fix_word_emb=False,
        emb_fix_bert_emb=False,
        emb_word_dropout=0,
        emb_rnn_dropout=0,
        dec_max_decoder_step=50,
        dec_use_copy=False,
        dec_use_coverage=False,
        dec_graph_pooling_strategy=None,
        dec_rnn_type="lstm",
        dec_tgt_emb_as_output_layer=False,
        dec_teacher_forcing_rate=1,
        dec_attention_type="uniform",
        dec_fuse_strategy="average",
        dec_node_type_num=None,
        dec_dropout=0,
        **kwargs
    ):
        super().__init__(
            vocab_model,
            emb_input_size,
            emb_hidden_size,
            embedding_style,
            graph_construction_name,
            gnn_direction_option,
            gnn_input_size,
            gnn_hidden_size,
            gnn_output_size,
            gnn,
            gnn_num_layers,
            dec_hidden_size,
            share_vocab,
            gnn_feat_drop,
            gnn_attn_drop,
            emb_fix_word_emb,
            emb_fix_bert_emb,
            emb_word_dropout,
            emb_rnn_dropout,
            dec_max_decoder_step,
            dec_use_copy,
            dec_use_coverage,
            dec_graph_pooling_strategy,
            dec_rnn_type,
            dec_tgt_emb_as_output_layer,
            dec_teacher_forcing_rate,
            dec_attention_type,
            dec_fuse_strategy,
            dec_node_type_num,
            dec_dropout,
            **kwargs
        )

    def _build_gnn_encoder(
        self,
        num_layers,
        input_size,
        hidden_size,
        output_size,
        feats_dropout,
        gnn_num_rels=80,
        gnn_num_bases=4,
        **kwargs
    ):
        self.gnn_encoder = RGCN(
            num_layers,
            input_size,
            hidden_size,
            output_size,
            num_rels=gnn_num_rels,
            num_bases=gnn_num_bases,
            dropout=feats_dropout,
        )