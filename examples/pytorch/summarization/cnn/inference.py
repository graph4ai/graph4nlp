"""
    The inference code.
    In this file, we will run the inference by using the prediction API \
        in the GeneratorInferenceWrapper.
    The GeneratorInferenceWrapper takes the raw inputs and produce the outputs.
"""
import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nltk.tokenize import word_tokenize

from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, to_cuda

from main import SumModel

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def all_to_cuda(data, device=None):
    if isinstance(data, torch.Tensor):
        data = to_cuda(data, device)
    elif isinstance(data, (list, dict)):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k], device)

    return data


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.use_copy = self.config["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.config["decoder_args"]["rnn_decoder_share"]["use_coverage"]

        self.stopper = EarlyStopping(
            os.path.join(
                "/raid/user8/graph4nlp", self.config["out_dir"], Constants._SAVED_WEIGHTS_FILE
            ),
            patience=self.config["patience"],
        )
        self._build_model()

    def _build_model(self):
        self.model = SumModel.load_checkpoint(self.stopper.save_model_path)
        self.model.vocab_model = self.model.vocab

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.config, model=self.model, beam_size=3, lower_case=True, tokenizer=word_tokenize
        )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        ret = self.inference_tool.predict(
            raw_contents=[
                """PARIS , France -LRB- CNN -RRB- -- Interpol is chasing more than 200 leads on
            the potential identity of a pedophile suspected of molesting young boys , just
            one day after launching a global manhunt . Interpol has launched a global appeal
            to find this man , accused of abusing young boys ."""
            ],
            batch_size=1,
        )
        print(ret)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-task_config", "--task_config", required=True, type=str, help="path to the config file"
    )
    parser.add_argument(
        "-g2s_config", "--g2s_config", required=True, type=str, help="path to the config file"
    )
    parser.add_argument("--grid_search", action="store_true", help="flag: grid search")
    args = vars(parser.parse_args())

    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def main(config):
    # configure
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if not config["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        config["device"] = torch.device("cuda" if config["gpu"] < 0 else "cuda:%d" % config["gpu"])
        cudnn.benchmark = True
        torch.cuda.manual_seed(config["seed"])
    else:
        config["device"] = torch.device("cpu")

    runner = ModelHandler(config)
    test_scores = runner.translate()

    return test_scores


if __name__ == "__main__":
    cfg = get_args()
    task_args = get_yaml_config(cfg["task_config"])
    g2s_args = get_yaml_config(cfg["g2s_config"])
    # load Graph2Seq template config
    g2s_template = get_basic_args(
        graph_construction_name=g2s_args["graph_construction_name"],
        graph_embedding_name=g2s_args["graph_embedding_name"],
        decoder_name=g2s_args["decoder_name"],
    )
    update_values(to_args=g2s_template, from_args_list=[g2s_args, task_args])
    print_config(g2s_template)
    main(g2s_template)
