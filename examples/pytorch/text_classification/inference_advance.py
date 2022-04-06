import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.trec import TrecDataset
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger

from .run_text_classifier import TextClassifier

torch.multiprocessing.set_sharing_strategy("file_system")


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        log_config = copy.deepcopy(config)
        del log_config["env_args"]["device"]
        self.logger = Logger(
            self.config["checkpoint_args"]["out_dir"],
            config=log_config,
            overwrite=True,
        )
        self.logger.write(self.config["checkpoint_args"]["out_dir"])
        self.model = TextClassifier.load_checkpoint(
            os.path.join(self.config["checkpoint_args"]["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.config["env_args"]["device"])
        self._build_dataloader()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = TrecDataset(
            root_dir=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["root_dir"],
            topology_subdir=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["topology_subdir"],
            graph_construction_name=self.config["model_args"]["graph_construction_name"],
            dynamic_init_graph_name=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_name", None),
            dynamic_init_topology_aux_args={"dummy_param": 0},
            pretrained_word_emb_name=self.config["preprocessing_args"]["pretrained_word_emb_name"],
            merge_strategy=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("merge_strategy", None),
            edge_strategy=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("edge_strategy", None),
            min_word_vocab_freq=self.config["preprocessing_args"]["min_word_freq"],
            word_emb_size=self.config["preprocessing_args"]["word_emb_size"],
            nlp_processor_args=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ].get("nlp_processor_args", None),
            seed=self.config["env_args"]["seed"],
            thread_number=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ].get("thread_number", None),
            for_inference=True,
            reused_vocab_model=self.model.vocab_model,
            reused_label_model=self.model.label_model,
        )

        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.config["env_args"]["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        self.config["num_classes"] = self.model.label_model.num_classes
        self.num_test = len(dataset.test)
        print("Test size: {}".format(self.num_test))
        self.logger.write("Test size: {}".format(self.num_test))

    def _build_evaluation(self):
        self.metric = Accuracy(["accuracy"])

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in dataloader:
                tgt = to_cuda(data["tgt_tensor"], self.config["env_args"]["device"])
                data["graph_data"] = data["graph_data"].to(self.config["env_args"]["device"])
                logits = self.model(data["graph_data"], require_loss=False)
                pred_collect.append(logits)
                gt_collect.append(tgt)

            pred_collect = torch.max(torch.cat(pred_collect, 0), dim=-1)[1].cpu()
            gt_collect = torch.cat(gt_collect, 0).cpu()
            score = self.metric.calculate_scores(ground_truth=gt_collect, predict=pred_collect)[0]

            pred_path = os.path.join(self.config["checkpoint_args"]["out_dir"], "pred.json")
            with open(pred_path, "w") as outf:
                json.dump(pred_collect.tolist(), outf)
                self.logger.write("Saved model predictions to {}".format(pred_path))

            return score

    def test(self):
        t0 = time.time()
        acc = self.evaluate(self.test_dataloader)
        dur = time.time() - t0
        print(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}".format(self.num_test, dur, acc)
        )
        self.logger.write(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}".format(self.num_test, dur, acc)
        )

        return acc


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
    t0 = time.time()

    test_acc = runner.test()

    runtime = time.time() - t0
    print("Total runtime: {:.2f}s".format(runtime))
    runner.logger.write("Total runtime: {:.2f}s\n".format(runtime))
    runner.logger.close()

    return test_acc


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
