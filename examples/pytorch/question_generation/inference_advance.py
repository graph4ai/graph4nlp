"""
    The advanced inference code.
    In this file, we will run the inference by writing the whole inference pipeline.
    Compared with the inference.py, it is more efficient. It will save the graphs \
        during inference, which support multi-processing when converting the raw inputs to graphs.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.squad import SQuADDataset
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.evaluation import BLEU, METEOR, ROUGE
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import all_to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger

from .main import QGModel  # noqa


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
        self._build_model()
        self._build_dataloader()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = SQuADDataset(
            root_dir=self.config["graph_construction_args"]["graph_construction_share"]["root_dir"],
            topology_subdir=self.config["graph_construction_args"]["graph_construction_share"][
                "topology_subdir"
            ],
            graph_name=self.config["graph_construction_args"]["graph_construction_share"][
                "graph_name"
            ],
            dynamic_init_graph_name=self.config["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_type", None),
            dynamic_init_topology_aux_args={"dummy_param": 0},
            pretrained_word_emb_name=self.config["pretrained_word_emb_name"],
            merge_strategy=self.config["graph_construction_args"]["graph_construction_private"][
                "merge_strategy"
            ],
            edge_strategy=self.config["graph_construction_args"]["graph_construction_private"][
                "edge_strategy"
            ],
            max_word_vocab_size=self.config["top_word_vocab"],
            min_word_vocab_freq=self.config["min_word_freq"],
            word_emb_size=self.config["word_emb_size"],
            share_vocab=self.config["share_vocab"],
            seed=self.config["seed"],
            thread_number=self.config["graph_construction_args"]["graph_construction_share"][
                "thread_number"
            ],
            port=self.config["graph_construction_args"]["graph_construction_share"]["port"],
            timeout=self.config["graph_construction_args"]["graph_construction_share"]["timeout"],
            for_inference=True,
            reused_vocab_model=self.model.vocab,
        )

        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        self.vocab = dataset.vocab_model
        self.num_test = len(dataset.test)
        print("Test size: {}".format(self.num_test))
        self.logger.write("Test size: {}".format(self.num_test))

    def _build_model(self):
        self.model = torch.load(
            os.path.join(self.config["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.config["device"])

    def _build_evaluation(self):
        self.metrics = {"BLEU": BLEU(n_grams=[1, 2, 3, 4]), "METEOR": METEOR(), "ROUGE": ROUGE()}

    def translate(self):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(self.test_dataloader):
                data = all_to_cuda(data, self.config["device"])
                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        data["graph_data"], self.vocab, device=self.config["device"]
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                batch_gd = self.model.encode_init_node_feature(data)
                prob = self.model.g2s.encoder_decoder_beam_search(
                    batch_gd, self.config["beam_size"], topk=1, oov_dict=oov_dict
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
                gt_collect.extend(data["tgt_text"])

            scores = self.evaluate_predictions(gt_collect, pred_collect)

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

    test_scores = runner.translate()
    print(test_scores)

    # print('Removed best saved model file to save disk space')
    # os.remove(runner.stopper.save_model_path)
    runtime = time.time() - t0
    print("Total runtime: {:.2f}s".format(time.time() - t0))
    runner.logger.write("Total runtime: {:.2f}s\n".format(runtime))
    runner.logger.close()

    return test_scores


def wordid2str(word_ids, vocab):
    ret = []
    assert len(word_ids.shape) == 2, print(word_ids.shape)
    for i in range(word_ids.shape[0]):
        id_list = word_ids[i, :]
        ret_inst = []
        for j in range(id_list.shape[0]):
            if id_list[j] == vocab.EOS or id_list[j] == vocab.PAD:
                break
            token = vocab.getWord(id_list[j])
            ret_inst.append(token)
        ret.append(" ".join(ret_inst))
    return ret


################################################################################
# ArgParse and Helper Functions #
################################################################################
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
        print("{} -->  {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    import multiprocessing
    import platform

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

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

    main(g2s_template)
