import argparse

from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_yaml",
        type=str,
        default="examples/pytorch/math_word_problem/mathqa/config/new_dynamic_graphsage_undirected.yaml",  # noqa
    )

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gpuid", type=int, default=1, help="which gpu to use. -1 = use CPU")
    parser.add_argument(
        "--seed", type=int, default=123, help="torch manual random number generator seed"
    )
    parser.add_argument("--init-weight", type=float, default=0.08, help="initailization weight")
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="number of full passes through the training data",
    )
    parser.add_argument("--min-freq", type=int, default=1, help="minimum frequency for vocabulary")
    parser.add_argument("--grad-clip", type=int, default=5, help="clip gradients at this value")

    # dataset config
    parser.add_argument("--batch-size", type=int, default=32, help="the size of one mini-batch")
    parser.add_argument("--share-vocab", type=bool, default=True, help="whether to share vocab")

    parser.add_argument("--pretrained_word_emb_name", type=str, default="6B", help="")
    parser.add_argument("--pretrained_word_emb_url", type=str, default=None, help="")
    parser.add_argument(
        "--pretrained_word_emb_cache_dir", type=str, default=".vector_cache", help=""
    )

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
