"""
    The inference code.
    In this file, we will run the inference by using the prediction API \
        in the GeneratorInferenceWrapper.
    The GeneratorInferenceWrapper takes the raw inputs and produce the outputs.
"""
import numpy as np
import torch

from graph4nlp.pytorch.datasets.jobs import tokenize_jobs
from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper_for_tree import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

import argparse
import random
import warnings

warnings.filterwarnings("ignore")


class Jobs:
    def __init__(self, opt=None):
        super(Jobs, self).__init__()
        self.opt = opt

        seed = self.opt["env_args"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.opt["env_args"]["gpuid"] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(self.opt["env_args"]["gpuid"]))

        self._build_model()

    def _build_model(self):
        self.model = Graph2Tree.load_checkpoint(
            self.opt["checkpoint_args"]["out_dir"], self.opt["checkpoint_args"]["checkpoint_name"]
        ).to(self.device)

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.opt, model=self.model, beam_size=2, lower_case=True, tokenizer=tokenize_jobs
        )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        ret = self.inference_tool.predict(raw_contents=["list job on platformid0"], batch_size=1)
        print(ret)


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-json_config",
        "--json_config",
        required=True,
        type=str,
        help="path to the json config file",
    )
    args = vars(parser.parse_args())

    return args


def print_config(config):
    import pprint

    print("**************** MODEL CONFIGURATION ****************")
    pprint.pprint(config)
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    import multiprocessing
    import platform

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    # print_config(config)

    runner = Jobs(opt=config)
    runner.translate()
