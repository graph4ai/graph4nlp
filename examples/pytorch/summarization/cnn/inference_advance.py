"""
    The advanced inference code.
    In this file, we will run the inference by writing the whole inference pipeline.
    Compared with the inference.py, it is more efficient. It will save the graphs \
        during inference, which support multi-processing when converting the raw inputs to graphs.
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.cnn import CNNDataset
from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import all_to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils.summarization_utils import wordid2str

import argparse
import copy
import os
import time

from .main import SumModel


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.use_copy = self.config["model_args"]["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.config["model_args"]["decoder_args"]["rnn_decoder_share"][
            "use_coverage"
        ]
        log_config = copy.deepcopy(config)
        del log_config["env_args"]["device"]
        self.logger = Logger(
            config["checkpoint_args"]["out_dir"],
            config=log_config,
            overwrite=True,
        )
        self.logger.write(config["checkpoint_args"]["out_dir"])
        self._build_model()
        self._build_dataloader()
        self._build_evaluation()

    def _build_dataloader(self):
        para_dic = {
            "root_dir": self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["root_dir"],
            "topology_subdir": self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["topology_subdir"],
            "word_emb_size": self.config["preprocessing_args"]["word_emb_size"],
            "merge_strategy": self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ]["merge_strategy"],
            "edge_strategy": self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ]["edge_strategy"],
            "graph_construction_name": self.config["model_args"]["graph_construction_name"],
            "share_vocab": self.config["preprocessing_args"]["share_vocab"],
            "min_word_vocab_freq": self.config["preprocessing_args"]["min_word_freq"],
            "thread_number": self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["thread_number"],
            "tokenizer": None,
            "for_inference": True,
            "reused_vocab_model": self.model.vocab,
        }

        inference_dataset = CNNDataset(**para_dic)

        self.inference_data_loader = DataLoader(
            inference_dataset.test,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.config["env_args"]["num_workers"],
            collate_fn=inference_dataset.collate_fn,
        )

        self.vocab = self.model.vocab
        self.num_test = len(inference_dataset.test)
        print("Test size: {}".format(self.num_test))
        self.logger.write("Test size: {}".format(self.num_test))

    def _build_model(self):
        self.model = SumModel.load_checkpoint(
            os.path.join(self.config["checkpoint_args"]["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.config["env_args"]["device"])

    def _build_evaluation(self):
        self.metrics = {"ROUGE": ROUGE()}

    def translate(self, dataloader, write2file=True):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                print(i)
                data = all_to_cuda(data, self.config["env_args"]["device"])
                data["graph_data"] = data["graph_data"].to(self.config["env_args"]["device"])

                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        data["graph_data"], self.vocab, device=self.config["env_args"]["device"]
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                batch_graph = self.model.g2s.graph_initializer(data["graph_data"])
                prob = self.model.g2s.encoder_decoder_beam_search(
                    batch_graph,
                    self.config["inference_args"]["beam_size"],
                    topk=1,
                    oov_dict=oov_dict,
                )

                pred_ids = (
                    torch.zeros(
                        len(prob),
                        self.config["model_args"]["decoder_args"]["rnn_decoder_private"][
                            "max_decoder_step"
                        ],
                    )
                    .fill_(ref_dict.EOS)
                    .to(self.config["env_args"]["device"])
                    .int()
                )
                for i, item in enumerate(prob):
                    item = item[0]
                    seq = [j.view(1, 1) for j in item]
                    seq = torch.cat(seq, dim=1)
                    pred_ids[i, : seq.shape[1]] = seq

                pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

                pred_collect.extend(pred_str)
                gt_collect.extend(data["output_str"])

            if write2file:
                with open(
                    "{}/{}_bs{}_pred.txt".format(
                        self.config["checkpoint_args"]["out_dir"],
                        self.config["checkpoint_args"]["out_dir"].split("/")[-1],
                        self.config["inference_args"]["beam_size"],
                    ),
                    "w+",
                ) as f:
                    for line in pred_collect:
                        f.write(line + "\n")

                with open(
                    "{}/{}_bs{}_gt.txt".format(
                        self.config["checkpoint_args"]["out_dir"],
                        self.config["checkpoint_args"]["out_dir"].split("/")[-1],
                        self.config["inference_args"]["beam_size"],
                    ),
                    "w+",
                ) as f:
                    for line in gt_collect:
                        f.write(line + "\n")

            scores = self.evaluate_predictions(gt_collect, pred_collect)

            return scores

    def test(self):
        t0 = time.time()
        scores = self.translate(self.inference_data_loader)
        dur = time.time() - t0
        format_str = "Test examples: {} | Time: {:.2f}s |  Test scores:".format(self.num_test, dur)
        format_str += self.metric_to_str(scores)
        print(format_str)
        self.logger.write(format_str)

        return scores

    def evaluate_predictions(self, ground_truth, predict):
        output = {}
        for name, scorer in self.metrics.items():
            score = scorer.calculate_scores(ground_truth=ground_truth, predict=predict)
            if name.upper() == "BLEU":
                for i in range(len(score[0])):
                    output["BLEU_{}".format(i + 1)] = score[0][i]
            else:
                output[name] = score[0]

        return output

    def metric_to_str(self, metrics):
        format_str = ""
        for k in metrics:
            format_str += " {} = {:0.5f},".format(k.upper(), metrics[k])

        return format_str[:-1]


def main(config):
    # configure
    np.random.seed(config["env_args"]["seed"])
    torch.manual_seed(config["env_args"]["seed"])

    if not config["env_args"]["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        config["env_args"]["device"] = torch.device(
            "cuda" if config["env_args"]["gpu"] < 0 else "cuda:%d" % config["env_args"]["gpu"]
        )
        cudnn.benchmark = True
        torch.cuda.manual_seed(config["env_args"]["seed"])
    else:
        config["env_args"]["device"] = torch.device("cpu")

    print("\n" + config["checkpoint_args"]["out_dir"])

    runner = ModelHandler(config)
    t0 = time.time()

    test_scores = runner.test()

    runtime = time.time() - t0
    print("Total runtime: {:.2f}s".format(time.time() - t0))
    runner.logger.write("Total runtime: {:.2f}s\n".format(runtime))
    runner.logger.close()

    return test_scores


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
        print("{} -->  {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    cfg = get_args()
    config = load_json_config(cfg["json_config"])

    main(config)
