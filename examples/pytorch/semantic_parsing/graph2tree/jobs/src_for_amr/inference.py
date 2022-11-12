"""
    The inference code.
    In this file, we will run the inference by using the prediction API \
        in the GeneratorInferenceWrapper.
    The GeneratorInferenceWrapper takes the raw inputs and produce the outputs.
"""
import argparse
import random
import warnings
import numpy as np
import torch
from utils import AMRDataItem

from graph4nlp.pytorch.datasets.mawps import MawpsDatasetForTree, tokenize_mawps
from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper_for_tree import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from examples.pytorch.amr_graph_construction.amr_graph_construction import AMRGraphConstruction
from utils import AMRDataItem, RGCNGraph2Tree, InferenceText2TreeDataset
warnings.filterwarnings("ignore")


class Mawps:
    def __init__(self, opt=None):
        super(Mawps, self).__init__()
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
        self.model = RGCNGraph2Tree.load_checkpoint(
            self.opt["checkpoint_args"]["out_dir"], self.opt["checkpoint_args"]["checkpoint_name"]
        ).to(self.device)

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.opt, 
            model=self.model, 
            beam_size=2, 
            lower_case=True, 
            tokenizer=tokenize_mawps, 
            dataset=InferenceText2TreeDataset,
            data_item=AMRDataItem,
            topology_builder=(AMRGraphConstruction if self.model.graph_construction_name == "amr" else None)
        )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        ret = self.inference_tool.predict(
            raw_contents=[
                "2 dogs are barking . 1 more dogs start to bark . how many dogs are barking"
            ],
            batch_size=1,
        )
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
    import platform
    import multiprocessing

    #if platform.system() == "Darwin":
    multiprocessing.set_start_method("spawn")

    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    # print_config(config)

    runner = Mawps(opt=config)
    runner.translate()
