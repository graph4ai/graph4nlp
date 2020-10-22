import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn",
                        help="")
    parser.add_argument("--topology_subdir", type=str, default='DependencyGraph4', help="")
    parser.add_argument('--word-emb-size', type=int, default=300, help='')
    parser.add_argument("--log-file", type=str, default="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/log/log4.txt")
    parser.add_argument("--checkpoint-save-path", type=str, default="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_dep4")
    # parser.add_argument("--checkpoint-save-path", type=str, default="/raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save")
    parser.add_argument('--hidden-size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='')
    parser.add_argument("--loss-display-step", type=int, default=100, help=' ')
    parser.add_argument("--eval-display-number", type=int, default=3, help="")
    parser.add_argument("--lr-start-decay-epoch", type=int, default=4, help="")
    parser.add_argument("--lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--lr-decay-per-epoch", type=int, default=5, help="")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="")
    parser.add_argument("--use_gpu", type=float, default=1, help="0 for don't use cuda, 1 for using cuda")
    parser.add_argument("--gpu", type=int, default=1, help="gpu id")
    parser.add_argument("--seed", type=int, default=1236, help="")

    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--gnn", type=str, default="Graphsage", help="")
    parser.add_argument("--direction_option", type=str, default="undirected", help="")
    parser.add_argument("--rnn_dropout", type=float, default=0.2, help="rnn dropout")
    parser.add_argument("--word_dropout", type=float, default=0.2, help="word dropout")
    cfg = parser.parse_args()
    return cfg

# screen -S ghn_code1 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir DependencyGraph5 --hidden-size 256 --lr-decay-per-epoch 1 --batch_size 20 --log-file examples/pytorch/summarization/cnn/log/log5.txt
# screen -S ghn_code2 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir DependencyGraph5 --hidden-size 256 --lr-decay-per-epoch 1 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log5_0.txt
# screen -S ghn_code3 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir DependencyGraph1 --hidden-size 256 --lr-decay-per-epoch 1 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log1.txt
# screen -S ghn_code1 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir DependencyGraph2 --hidden-size 256 --lr-decay-per-epoch 1 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log2.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_dep2
# screen -S ghn_code2 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir DependencyGraph2 --hidden-size 256 --lr-decay-per-epoch 1 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log2_0.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_dep2_0 --learning-rate 0.0005
# screen -S ghn_code3 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir DependencyGraph_10 --hidden-size 256 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log_10.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_dep_10 --lr-decay-rate  0.8
# screen -S ghn_code2 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir DependencyGraph_10 --hidden-size 256 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log_10_3.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_dep_10_3 --lr-decay-rate 0.8
# screen -S ghn_code3 python -m examples.pytorch.summarization.cnn.main --word-emb-size 128 --topology_subdir LinearGraph --hidden-size 256 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log_lg_0.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_lg_0 --lr-decay-rate  0.8
# screen -S ghn_code3 python -m examples.pytorch.summarization.cnn.main --word-emb-size 300 --topology_subdir DependencyGraph_seq2seq --hidden-size 256 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log_s2s.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_dep_s2s --lr-decay-rate 0.8 --gpu 1 --learning-rate 0.01
# screen -S ghn_code2 python -m examples.pytorch.summarization.cnn.main --word-emb-size 300 --topology_subdir DependencyGraph_seq2seq --hidden-size 256 --batch_size 30 --log-file examples/pytorch/summarization/cnn/log/log_s2s.txt --checkpoint-save-path /raid/ghn/graph4nlp/examples/pytorch/summarization/cnn/save_dep_s2s --lr-decay-rate 0.8 --learning-rate 0.01 --gpu 2
