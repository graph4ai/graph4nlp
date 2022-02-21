import argparse
import os
from typing import List
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from nltk.tokenize import word_tokenize

from graph4nlp.pytorch.data.dataset import Text2LabelDataItem, Text2LabelDataset
from graph4nlp.pytorch.inference_wrapper.classifier_inference_wrapper import (
    ClassifierInferenceWrapper,
)
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

from .run_text_classifier import TextClassifier

torch.multiprocessing.set_sharing_strategy("file_system")


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        self.model = TextClassifier.load_checkpoint(
            os.path.join(self.config["checkpoint_args"]["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.config["env_args"]["device"])

        self.inference_tool = ClassifierInferenceWrapper(
            cfg=self.config,
            model=self.model,
            label_names=self.model.label_model.le.classes_.tolist(),
            dataset=Text2LabelDataset,
            data_item=Text2LabelDataItem,
            lower_case=True,
            tokenizer=word_tokenize,
        )

    @torch.no_grad()
    def predict(self, raw_contents: List[str], batch_size=1):
        """raw_contents contains a list of text"""
        self.model.eval()
        ret = self.inference_tool.predict(raw_contents=raw_contents, batch_size=batch_size)
        return ret


def main(config):
    # configure
    np.random.seed(config["env_args"]["seed"])
    torch.manual_seed(config["env_args"]["seed"])

    if not config["env_args"]["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        config["env_args"]["device"] = torch.device(
            "cuda" if config["env_args"]["gpu"] < 0 else "cuda:%d" % config["env_args"]["gpu"]
        )
        torch.cuda.manual_seed(config["env_args"]["seed"])
        torch.cuda.manual_seed_all(config["env_args"]["seed"])
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        config["env_args"]["device"] = torch.device("cpu")

    print("\n" + config["checkpoint_args"]["out_dir"])

    runner = ModelHandler(config)
    ret = runner.predict(["How far is it from Denver to Aspen ?"])
    print(f"output: {ret}")


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
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    import platform
    import multiprocessing

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    print_config(config)

    main(config)
