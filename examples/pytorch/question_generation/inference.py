"""
    The inference code.
    In this file, we will run the inference by using the prediction API \
        in the GeneratorInferenceWrapper.
    The GeneratorInferenceWrapper takes the raw inputs and produce the outputs.
"""
import argparse
import os
from typing import List, Tuple
import numpy as np
import torch
from nltk.tokenize import word_tokenize

from graph4nlp.pytorch.data.dataset import DoubleText2TextDataItem, DoubleText2TextDataset
from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

from .main import QGModel  # noqa


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
        self.model = torch.load(
            os.path.join(self.config["checkpoint_args"]["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.device)

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.config,
            model=self.model,
            dataset=DoubleText2TextDataset,
            data_item=DoubleText2TextDataItem,
            beam_size=self.config["inference_args"]["beam_size"],
            topk=1,
            lower_case=True,
            tokenizer=word_tokenize,
            share_vocab=True,
        )

    @torch.no_grad()
    def translate(self, raw_contents: List[Tuple[str]], batch_size=1):
        """raw_contents contains a list of (context, answer) pairs"""
        self.model.eval()
        ret = self.inference_tool.predict(raw_contents=raw_contents, batch_size=batch_size)
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
    import platform
    import multiprocessing

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    cfg = get_args()
    config = load_json_config(cfg["json_config"])

    runner = ModelHandler(config)
    ret = runner.translate(
        [
            (
                "Beijing is the capital of China",
                "Beijing",
            )
        ],
        batch_size=1,
    )
    print(f"output: {ret}")
