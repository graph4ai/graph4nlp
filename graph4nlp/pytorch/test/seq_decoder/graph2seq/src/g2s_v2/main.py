import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import yaml
import torch
import numpy as np
# import far_ho as far
from collections import OrderedDict

from graph4nlp.pytorch.test.seq_decoder.graph2seq.src.g2s_v2.core.model_handler import ModelHandler

import warnings

warnings.filterwarnings('ignore')

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(119)
    model = ModelHandler(config)
    model.train()
    model.test()


### ghn ###
def eval(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.model.init_saved_network(config['out_dir'])
    # model.train()
    model.test()


def grid_search_main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    grid_search_hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            grid_search_hyperparams.append(k)

    best_config = None
    best_metric = None
    best_score = -1
    configs = grid(config)
    for cnf in configs:
        print('\n')
        for k in grid_search_hyperparams:
            cnf['out_dir'] += '_{}_{}'.format(k, cnf[k])
        print(cnf['out_dir'])
        model = ModelHandler(cnf)
        dev_metrics = model.train()
        test_metrics = model.test()
        if best_score < test_metrics[cnf['eary_stop_metric']]:
            best_score = test_metrics[cnf['eary_stop_metric']]
            best_config = cnf
            best_metric = test_metrics
            print('Found a better configuration: {}'.format(best_score))

    print('\nBest configuration:')
    for k in grid_search_hyperparams:
        print('{}: {}'.format(k, best_config[k]))

    print('Best Dev score: {}'.format(best_score))

################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.safe_load(setting)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default="/home/shiina/shiina/lib/graph4nlp/graph4nlp/pytorch/test/seq_decoder/graph2seq/src/g2s_v2/config/JOBS/JOBS_graph2seq_wo_pretrain_1234.yml", type=str, help='path to the config file')
    parser.add_argument('--grid_search', action='store_true', help='flag: grid search')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [far.utils.merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    if cfg['grid_search']:
        grid_search_main(config)
    else:
        main(config)
        # eval(config)
