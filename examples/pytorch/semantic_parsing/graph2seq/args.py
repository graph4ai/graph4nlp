import argparse, yaml


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def get_yaml_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_yaml", type=str,
                        default="examples/pytorch/semantic_parsing/graph2seq/config/dependency.yaml", help="")
    parser.add_argument("--use_copy", type=bool, default=True, help="whether use copy mechanism")
    parser.add_argument("--use_coverage", type=bool, default=True, help="whether use coverage mechanism")
    parser.add_argument("--graph_pooling_strategy", type=str, default=None,
                        help="The strategy of graph pooling for initial state of RNN, expected in (None, 'mean'"
                             ",'max', 'min')")
    parser.add_argument("--attention_type", type=str, default="sep_diff_encoder_type",
                        help="attention strategy")
    parser.add_argument("--fuse_strategy", type=str, default="concatenate", help="fuse strategy")
    parser.add_argument("--decoder_length", type=int, default=50, help="the maximum length of output text")
    parser.add_argument('--word-emb-size', type=int, default=300, help='')
    parser.add_argument("--log-file", type=str, default="examples/pytorch/semantic_parsing/graph2seq/log/ggnn.txt")
    parser.add_argument("--checkpoint-save-path", type=str, default="examples/pytorch/semantic_parsing/graph2seq/save")
    parser.add_argument('--hidden-size', type=int, default=500, help='')
    # dropout
    parser.add_argument('--emb-dropout', type=float, default=0.2, help='')
    parser.add_argument('--feats-dropout', type=float, default=0.2, help='')
    parser.add_argument('--rnn-dropout', type=float, default=0.3, help='')

    parser.add_argument('--learning-rate', type=float, default=1e-3, help='')
    parser.add_argument("--loss-display-step", type=int, default=3, help=' ')
    parser.add_argument("--eval-display-number", type=int, default=3, help="")
    parser.add_argument("--lr-start-decay-epoch", type=int, default=20, help="")
    parser.add_argument("--lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--lr-decay-per-epoch", type=int, default=5, help="")
    parser.add_argument("--min-lr", type=float, default=1e-3, help="")
    parser.add_argument("--use-gpu", type=float, default=1, help="0 for don't use cuda, 1 for using cuda")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=1236, help="")

    # dataset config
    parser.add_argument("--batch_size", type=int, default=24, help="the size of one mini-batch")
    parser.add_argument("--dynamic_graph_type", type=str, default="node_emb_refined",
                        help="graph type, expected in (None, 'node_emb', 'node_emb_refined')")
    parser.add_argument("--graph-type", type=str, default="dynamic",
                        help="graph type, expected in ('static', 'dynamic'")
    parser.add_argument("--root-dir", type=str, default="graph4nlp/pytorch/test/dataset/jobs",
                        help="the root dir of dataset")
    parser.add_argument("--topology-subdir", type=str, default="NodeEmbGraphRefine",
                        help="the dir to store the processed data")
    parser.add_argument("--edge-strategy", type=str, default="homogeneous", help="the strategy for edge,"
                                                                             "expected in ('homogeneous', 'as_node')")
    parser.add_argument("--share-vocab", type=bool, default=True, help="whether to share vocab")
    parser.add_argument("--dynamic_init_graph_type", type=str, default=None, help="")
    parser.add_argument("--init_graph_type", type=str, default=None, help="")
    parser.add_argument("--merge_strategy", type=str, default="tailhead", help="")
    parser.add_argument("--val_split_ratio", type=float, default=0, help="")
    parser.add_argument("--pretrained_word_emb_file", type=str,
                        default='/home/shiina/shiina/lib/graph4nlp/.vector_cache/glove.6B.300d.txt', help="")

    # gnn config
    parser.add_argument("--gnn", type=str, default="GraphSage", help="the gnn algorithm choice in ('GAT', 'GGNN', 'GraphSage', 'GCN')")
    parser.add_argument("--gnn-direction", type=str, default="bi_sep", help="gnn direction, expected in "
                                                                                "('undirected', 'bi_sep', 'bi_fuse')")
    #beam search
    parser.add_argument("--beam-size", type=int, default=4, help="the beam size of beam search")

    cfg = parser.parse_args()

    dataset_args = get_yaml_config(cfg.dataset_yaml)
    update_values(dataset_args, vars(cfg))
    return cfg
