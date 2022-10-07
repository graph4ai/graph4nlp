"""
    The inference code.
    In this file, we will run the inference by using the prediction API \
        in the GeneratorInferenceWrapper.
    The GeneratorInferenceWrapper takes the raw inputs and produce the outputs.
"""
import argparse
import os
from typing import List
import numpy as np
import torch
from nltk.tokenize import word_tokenize

from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

from .main import SumModel


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self._build_device(self.config)
        self._build_model()

    def _build_device(self, config):
        seed = config["env_args"]["seed"]
        np.random.seed(seed)
        if not config["env_args"]["no_cuda"] and torch.cuda.is_available():
            print("[ Using CUDA ]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = torch.device(
                "cuda" if config["env_args"]["gpu"] < 0 else "cuda:%d" % config["env_args"]["gpu"]
            )
        else:
            print("[ Using CPU ]")
            device = torch.device("cpu")
        self.device = device

    def _build_model(self):
        self.model = SumModel.load_checkpoint(
            os.path.join(self.config["checkpoint_args"]["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.device)

        self.model.vocab_model = self.model.vocab

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.config,
            model=self.model,
            beam_size=self.config["inference_args"]["beam_size"],
            topk=1,
            lower_case=True,
            tokenizer=word_tokenize,
            share_vocab=True,
        )

    @torch.no_grad()
    def translate(self, raw_contents: List[str], batch_size=1):
        self.model.eval()
        ret = self.inference_tool.predict(
            raw_contents=raw_contents,
            batch_size=batch_size,
        )
        return ret


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


if __name__ == "__main__":
    cfg = get_args()
    config = load_json_config(cfg["json_config"])

    runner = ModelHandler(config)
    ret = runner.translate(
        [
            """PARIS , France -LRB- CNN -RRB- -- Interpol is chasing more than 200 leads on
            the potential identity of a pedophile suspected of molesting young boys , just
            one day after launching a global manhunt . Interpol has launched a global appeal
            to find this man , accused of abusing young boys ."""
        ],
        batch_size=1,
    )
    print(f"output: {ret}")
