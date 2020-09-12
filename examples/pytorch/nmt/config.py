import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/home/shiina/shiina/lib/dataset/news-commentary-v11/de-en",
                        help="")
    parser.add_argument("--topology_subdir", type=str, default='DependencyGraph', help="")
    parser.add_argument('--word-emb-size', type=int, default=300, help='')
    parser.add_argument("--log-file", type=str, default="examples/pytorch/nmt/log/log.txt")
    parser.add_argument("--checkpoint-save-path", type=str, default="examples/pytorch/nmt/save")
    parser.add_argument('--hidden-size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='')
    parser.add_argument("--loss-display-step", type=int, default=100, help=' ')
    parser.add_argument("--eval-display-number", type=int, default=3, help="")
    parser.add_argument("--lr-start-decay-epoch", type=int, default=4, help="")
    parser.add_argument("--lr-decay-rate", type=float, default=0.8)
    parser.add_argument("--lr-decay-per-epoch", type=int, default=5, help="")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="")
    parser.add_argument("--use_gpu", type=float, default=1, help="0 for don't use cuda, 1 for using cuda")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=1236, help="")

    parser.add_argument("--batch_size", type=int, default=30, help="batch size")
    parser.add_argument("--gnn", type=str, default="Graphsage", help="")
    parser.add_argument("--direction_option", type=str, default="undirected", help="")
    parser.add_argument("--rnn_dropout", type=float, default=0.2, help="rnn dropout")
    parser.add_argument("--word_dropout", type=float, default=0.2, help="word dropout")
    cfg = parser.parse_args()
    return cfg
