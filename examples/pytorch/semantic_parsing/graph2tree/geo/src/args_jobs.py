import argparse

from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_yaml", type=str,
                        default="examples/pytorch/semantic_parsing/graph2tree/config/new_dynamic_graphsage.yaml", 
                        help="which graph construction method and gnn type you want to use.")

    parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    parser.add_argument('-seed', type=int, default=123, help='torch manual random number generator seed')

    parser.add_argument('-data_dir', type=str,
                        default='/home/lishucheng/Graph4AI/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/data/jobs', help='data path')

    parser.add_argument('-checkpoint_dir', type=str,
                        default='/home/lishucheng/Graph4AI/graph4nlp/examples/pytorch/semantic_parsing/graph2tree/checkpoint_dir_jobs',
                        help='output directory where checkpoints get written')

    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-init_weight', type=float, default=0.08, help='initailization weight')
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-max_epochs', type=int, default=150,help='number of full passes through the training data')
    parser.add_argument('-min_freq', type=int, default=1,help='minimum frequency for vocabulary')
    parser.add_argument('-grad_clip', type=int, default=5, help='clip gradients at this value')

    cfg = parser.parse_args()

    our_args = get_yaml_config(cfg.dataset_yaml)
    template = get_basic_args(graph_construction_name=our_args["graph_construction_name"],
                              graph_embedding_name=our_args["graph_embedding_name"],
                              decoder_name=our_args["decoder_name"])
    update_values(to_args=template, from_args_list=[our_args, vars(cfg)])
    return template
