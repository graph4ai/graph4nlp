import argparse

def get_args():
    main_arg_parser = argparse.ArgumentParser(description="parser")

    main_arg_parser.add_argument('-gpuid', type=int, default=3, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-seed', type=int, default=1234, help='torch manual random number generator seed')
    main_arg_parser.add_argument('-use_copy', type=int, default=1, help='whether use copy mechanism')
    main_arg_parser.add_argument('-data_dir', type=str, default='/home/lishucheng/Graph4AI/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/geo/geo_data/', help='data path')

    main_arg_parser.add_argument('-gnn_type', type=str, default="SAGE")
    main_arg_parser.add_argument('-gat_head', type=str, default="1")
    main_arg_parser.add_argument('-sage_aggr', type=str, default="lstm")
    main_arg_parser.add_argument('-attn_type', type=str, default="uniform")
    main_arg_parser.add_argument('-use_sibling', type=int, default=0)
    main_arg_parser.add_argument('-use_share_vocab', type=int, default=1)
    main_arg_parser.add_argument('-K', type=int, default=1)

    main_arg_parser.add_argument('-enc_emb_size', type=int, default=300)
    main_arg_parser.add_argument('-tgt_emb_size', type=int, default=300)

    main_arg_parser.add_argument('-enc_hidden_size', type=int, default=300)
    main_arg_parser.add_argument('-dec_hidden_size', type=int, default=300)

    # DynamicGraph_node_emb_refined, DynamicGraph_node_emb, ConstituencyGraph
    main_arg_parser.add_argument('-graph_construction_type', type=str, default="DynamicGraph_node_emb")

    # "None, line, dependency, constituency"
    main_arg_parser.add_argument('-dynamic_init_graph_type', type=str, default="constituency")
    main_arg_parser.add_argument('-batch_size', type=int, default=20)
    main_arg_parser.add_argument('-dropout_for_word_embedding', type=float, default=0.1)
    main_arg_parser.add_argument('-dropout_for_encoder', type=float, default=0)
    main_arg_parser.add_argument('-dropout_for_decoder', type=float, default=0.1)

    main_arg_parser.add_argument('-direction_option', type=str, default="undirected")
    main_arg_parser.add_argument('-beam_size', type=int, default=2)

    main_arg_parser.add_argument('-max_dec_seq_length', type=int, default=100)
    main_arg_parser.add_argument('-max_dec_tree_depth', type=int, default=30)

    main_arg_parser.add_argument('-teacher_force_ratio', type=float, default=1.0)
    main_arg_parser.add_argument('-init_weight', type=float, default=0.08, help='initailization weight')
    main_arg_parser.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate')
    main_arg_parser.add_argument('-weight_decay', type=float, default=0)
    main_arg_parser.add_argument('-max_epochs', type=int, default=200,help='number of full passes through the training data')
    main_arg_parser.add_argument('-min_freq', type=int, default=1,help='minimum frequency for vocabulary')
    main_arg_parser.add_argument('-grad_clip', type=int, default=5, help='clip gradients at this value')

    args = main_arg_parser.parse_args()
    return args