import argparse

from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="name of this run")
    parser.add_argument(
        "--dataset_yaml", type=str, default="examples/pytorch/nmt/config/dynamic_gcn.yaml", help=""
    )

    parser.add_argument("--word-emb-size", type=int, default=300, help="")
    parser.add_argument("--log-dir", type=str, default="examples/pytorch/nmt/log/")
    parser.add_argument("--checkpoint-save-path", type=str, default="examples/pytorch/nmt/save")

    parser.add_argument("--learning-rate", type=float, default=0.001, help="")
    parser.add_argument("--loss-display-step", type=int, default=50, help=" ")
    parser.add_argument("--eval-display-number", type=int, default=1, help="")
    parser.add_argument("--warmup_steps", default=300, type=int, help="warm up setting")
    parser.add_argument("--max_steps", default=28000, type=int, help="max updating steps")
    parser.add_argument("--lr-start-decay-epoch", type=int, default=100, help="")
    parser.add_argument("--lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--lr-decay-per-epoch", type=int, default=3, help="")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="")
    parser.add_argument(
        "--use-gpu", type=float, default=1, help="0 for don't use cuda, 1 for using cuda"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=1236, help="")

    # dataset config
    parser.add_argument("--batch_size", type=int, default=128, help="the size of one mini-batch")
    parser.add_argument("--share-vocab", type=bool, default=False, help="whether to share vocab")
    parser.add_argument("--val_split_ratio", type=float, default=0, help="")

    parser.add_argument("--beam-size", type=int, default=4, help="the beam size of beam search")

    cfg = parser.parse_args()

    our_args = get_yaml_config(cfg.dataset_yaml)
    template = get_basic_args(
        graph_construction_name=our_args["graph_construction_name"],
        graph_embedding_name=our_args["graph_embedding_name"],
        decoder_name=our_args["decoder_name"],
    )
    update_values(to_args=template, from_args_list=[our_args, vars(cfg)])
    return template
