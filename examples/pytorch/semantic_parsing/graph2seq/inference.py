"""
    The inference code.
    In this file, we will run the inference by using the prediction API \
        in the GeneratorInferenceWrapper.
    The GeneratorInferenceWrapper takes the raw inputs and produce the outputs.
"""
import numpy as np
import torch
from nltk.tokenize import word_tokenize

from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

from args import get_args


class Jobs:
    def __init__(self, opt):
        super(Jobs, self).__init__()
        self.opt = opt
        self._build_device(self.opt)
        self._build_model()

    def _build_device(self, opt):
        seed = opt["env_args"]["seed"]
        np.random.seed(seed)
        if opt["env_args"]["use_gpu"] != 0 and torch.cuda.is_available():
            print("[ Using CUDA ]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = torch.device("cuda" if opt["env_args"]["gpuid"] < 0 else
                                  "cuda:%d" % opt["env_args"]["gpuid"])
        else:
            print("[ Using CPU ]")
            device = torch.device("cpu")
        self.device = device

    def _build_model(self):
        self.model = Graph2Seq.load_checkpoint(
            self.opt["checkpoint_args"]["checkpoint_save_path"], "best.pt").to(self.device)

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.opt, model=self.model, beam_size=self.opt["inference_args"]["beam_size"],
            lower_case=True, tokenizer=word_tokenize
        )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        ret = self.inference_tool.predict(raw_contents=["list job on platformid0"], batch_size=1)
        print(ret)


if __name__ == "__main__":
    opt = get_args()
    config = load_json_config(opt["json_config"])
    runner = Jobs(config)
    runner.translate()
