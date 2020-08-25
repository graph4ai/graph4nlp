import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-emb-size', type=int, default=300, help='')
    parser.add_argument("--log-file", type=str, default="examples/pytorch/semantic_parsing/jobs/log/tuning.txt")
    parser.add_argument("--checkpoint-save-path", type=str, default="examples/pytorch/semantic_parsing/jobs/save")
    parser.add_argument('--hidden-size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='')
    parser.add_argument("--loss-display-step", type=int, default=3, help=' ')
    parser.add_argument("--eval-display-number", type=int, default=3, help="")
    parser.add_argument("--lr-start-decay-epoch", type=int, default=20, help="")
    parser.add_argument("--lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--lr-decay-per-epoch", type=int, default=5, help="")
    parser.add_argument("--min-lr", type=float, default=1e-3, help="")
    parser.add_argument("--use_gpu", type=float, default=1, help="0 for don't use cuda, 1 for using cuda")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=1236, help="")
    cfg = parser.parse_args()
    return cfg
