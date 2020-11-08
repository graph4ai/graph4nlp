from graph4nlp.pytorch.models.graph2seq import Graph2Seq


def get_model(opt, vocab_model, device):
    # if opt.graph_type in ["node_emb", "node_emb_refined"]:
    #     extra_params = {
    #         "sim_metric_type": opt.sim_metric_type,
    #         "num_heads": opt.num_heads,
    #         "epsilon_neigh": opt.epsilon_neigh,
    #         "smoothness_ratio": opt.smoothness_ratio,
    #         "connectivity_ratio": opt.connectivity_ratio,
    #         "sparsity_ratio": opt.sparsity_ratio,
    #         "alpha_fusion": opt.alpha_fusion if opt.graph_type == "node_emb_refined" else None
    #     }
    # else:
    #     extra_params = {}
    model = Graph2Seq.from_args(opt, vocab_model, device)

    # model = Graph2Seq(vocab_model=vocab_model,
    #                   encoder_hidden_size=opt.hidden_size, decoder_hidden_size=opt.hidden_size,
    #                   graph_type=opt.graph_type, direction_option=opt.direction_option, gnn=opt.gnn,
    #                   gnn_layer_number=opt.gnn_layer_number, embedding_style=embedding_style,
    #                   max_decoder_depth=opt.decoder_length, use_copy=opt.use_copy, use_coverage=opt.use_coverage,
    #                   graph_pooling_strategy=opt.graph_pooling_strategy, tgt_emb_as_output_layer=True,
    #                   attention_type="sep_diff_encoder_type", fuse_strategy=opt.fuse_strategy,
    #                   feats_dropout=opt.feats_dropout, attn_dropout=opt.attn_dropout, rnn_dropout=opt.rnn_dropout,
    #
    #                   # # embedding parameters
    #                   # sim_metric_type=opt.sim_metric_type, num_heads=opt.num_heads, epsilon_neigh=opt.epsilon_neigh,
    #                   # smoothness_ratio=opt.smoothness_ratio, connectivity_ratio=opt.connectivity_ratio,
    #                   # sparsity_ratio=opt.sparsity_ratio,
    #                   # alpha_fusion=opt.alpha_fusion if opt.graph_type == "node_emb_refined" else None,
    #
    #                   # gnn parameters
    #                   gat_heads=[2, 2, 1] if opt.gnn == "GAT" else None,
    #                   device=device, **extra_params)
    return model
