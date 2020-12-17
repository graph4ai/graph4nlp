import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="examples/pytorch/summarization/cnn",
                        help="")
    parser.add_argument("--topology_subdir", type=str, default='DependencyGraph_3w', help="")
    parser.add_argument('--word-emb-size', type=int, default=128, help='')
    parser.add_argument("--log-file", type=str, default="examples/pytorch/summarization/cnn/log/log_dep_30k.txt")
    parser.add_argument("--checkpoint-save-path", type=str, default="examples/pytorch/summarization/cnn/save_dep_30k")
    parser.add_argument('--hidden-size', type=int, default=512, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='')
    parser.add_argument("--loss-display-step", type=int, default=100, help=' ')
    parser.add_argument("--eval-display-number", type=int, default=3, help="")
    parser.add_argument("--lr-start-decay-epoch", type=int, default=4, help="")
    parser.add_argument("--lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--lr-decay-per-epoch", type=int, default=5, help="")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="")
    parser.add_argument("--use_gpu", type=float, default=1, help="0 for don't use cuda, 1 for using cuda")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=1236, help="")

    parser.add_argument("--use_copy", type=bool, default=False, help="")
    parser.add_argument("--use_coverage", type=bool, default=False, help="")

    parser.add_argument("--batch_size", type=int, default=30, help="batch size")
    parser.add_argument("--gnn", type=str, default="GraphSage", help="")
    parser.add_argument("--direction_option", type=str, default="undirected", help="")
    parser.add_argument("--rnn_dropout", type=float, default=0.2, help="rnn dropout")
    parser.add_argument("--word_dropout", type=float, default=0.2, help="word dropout")
    cfg = parser.parse_args()
    return cfg

# screen -S ghn_code1 python -m examples.pytorch.summarization.cnn.main_g2s --word-emb-size 128 --topology_subdir DependencyGraph_3w --hidden-size 512 --batch_size 40 --log-file examples/pytorch/summarization/cnn/log/log_g2s_30w_copy.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_g2s_3w_copy --lr-decay-rate 0.8 --learning-rate 1e-3 --use_copy True --use_coverage True
# screen -S ghn_code2 python -m examples.pytorch.summarization.cnn.main_g2s --word-emb-size 128 --topology_subdir DependencyGraph_9w --hidden-size 512 --batch_size 40 --log-file examples/pytorch/summarization/cnn/log/log_g2s_9w_copy.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_g2s_9w_copy --lr-decay-rate 0.8 --learning-rate 1e-3 --use_copy True --use_coverage True

# --root_dir
# .
# --word-emb-size
# 128
# --topology_subdir
# Seq2seq_30k_t
# --hidden-size
# 512
# --batch_size
# 40
# --log-file
# log/log_s2s_30k_copy3.txt
# --checkpoint-save-path
# save_s2s_30k_copy3
# --lr-decay-rate
# 0.8
# --learning-rate
# 0.001
# --use_copy
# True
# --use_coverage
# True
