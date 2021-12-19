import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.cnn import CNNDataset
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils.summarization_utils import wordid2str

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
        self.logger = Logger(
            config["out_dir"],
            config={k: v for k, v in config.items() if k != "device"},
            overwrite=True,
        )
        self.logger.write(config["out_dir"])

        self.stopper = EarlyStopping(
            os.path.join(
                "/raid/user8/graph4nlp", self.config["out_dir"], Constants._SAVED_WEIGHTS_FILE
            ),
            patience=self.config["patience"],
        )
        self._build_model()
        self._build_dataloader()

        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):

        para_dic = {
            "root_dir": self.config["graph_construction_args"]["graph_construction_share"][
                "root_dir"
            ],
            "word_emb_size": self.config["word_emb_size"],
            "topology_subdir": self.config["graph_construction_args"]["graph_construction_share"][
                "topology_subdir"
            ],
            "edge_strategy": self.config["graph_construction_args"]["graph_construction_private"][
                "edge_strategy"
            ],
            "graph_name": self.config["graph_construction_args"]["graph_construction_share"][
                "graph_name"
            ],
            "share_vocab": self.config["share_vocab"],
            "min_word_vocab_freq": self.config["min_word_freq"],
            "thread_number": self.config["graph_construction_args"]["graph_construction_share"][
                "thread_number"
            ],
            "port": self.config["graph_construction_args"]["graph_construction_share"]["port"],
            "timeout": self.config["graph_construction_args"]["graph_construction_share"][
                "timeout"
            ],
            "tokenizer": None,
            "for_inference": 1,
            "reused_vocab_model": self.model.vocab,
        }

        inference_dataset = CNNDataset(**para_dic)

        self.inference_data_loader = DataLoader(
            inference_dataset.test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=inference_dataset.collate_fn,
        )

        self.vocab = self.model.vocab
        self.src_vocab = self.vocab.in_word_vocab
        self.tgt_vocab = self.vocab.out_word_vocab

        self.num_test = len(inference_dataset.test)

    def _build_model(self):
        self.model = SumModel.load_checkpoint(self.stopper.save_model_path)
        # self.logger.write(str(self.model))

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config["lr"])
        self.stopper = EarlyStopping(
            os.path.join(self.config["out_dir"], Constants._SAVED_WEIGHTS_FILE),
            patience=self.config["patience"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config["lr_reduce_factor"],
            patience=self.config["lr_patience"],
            verbose=True,
        )

    def _build_evaluation(self):
        self.metrics = {"ROUGE": ROUGE()}

    def translate(self, dataloader, write2file=True):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                print(i)
                data = all_to_cuda(data, self.config["device"])
                data["graph_data"] = data["graph_data"].to(self.config["device"])

                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        data["graph_data"], self.vocab, device=self.config["device"]
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab
                batch_graph = self.model.g2s.graph_initializer(data["graph_data"])
                prob = self.model.g2s.encoder_decoder_beam_search(
                    batch_graph, self.config["beam_size"], topk=1, oov_dict=oov_dict
                )

                pred_ids = (
                    torch.zeros(
                        len(prob),
                        self.config["decoder_args"]["rnn_decoder_private"]["max_decoder_step"],
                    )
                    .fill_(ref_dict.EOS)
                    .to(self.config["device"])
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
                        self.config["out_dir"],
                        self.config["out_dir"].split("/")[-1],
                        self.config["beam_size"],
                    ),
                    "w+",
                ) as f:
                    for line in pred_collect:
                        f.write(line + "\n")

                with open(
                    "{}/{}_bs{}_gt.txt".format(
                        self.config["out_dir"],
                        self.config["out_dir"].split("/")[-1],
                        self.config["beam_size"],
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

    print("\n" + config["out_dir"])

    runner = ModelHandler(config)
    t0 = time.time()

    test_scores = runner.test()

    runtime = time.time() - t0
    print("Total runtime: {:.2f}s".format(time.time() - t0))
    runner.logger.write("Total runtime: {:.2f}s\n".format(runtime))
    runner.logger.close()

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
