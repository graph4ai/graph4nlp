"""
    The advanced inference code.
    In this file, we will run the inference by writing the whole inference pipeline.
    Compared with the inference.py, it is more efficient. It will save the graphs \
        during inference, which support multi-processing when converting the raw inputs to graphs.
"""
import argparse
import copy
import random
import time
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.mawps import MawpsDatasetForTree
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

from evaluation import SolutionMatch

warnings.filterwarnings("ignore")

DUMMY_STR = "<TBD>"


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

        self.use_copy = self.opt["model_args"]["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_share_vocab = self.opt["model_args"]["graph_construction_args"][
            "graph_construction_share"
        ]["share_vocab"]
        self.data_dir = self.opt["inference_args"]["inference_data_dir"]

        self._build_model()
        self._build_dataloader()
        self._build_evaluation()

    def _build_dataloader(self):
        para_dic = {
            "root_dir": self.data_dir,
            "word_emb_size": self.opt["model_args"]["graph_initialization_args"]["input_size"],
            "topology_subdir": self.opt["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["topology_subdir"],
            "edge_strategy": self.opt["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ]["edge_strategy"],
            "graph_construction_name": self.opt["model_args"]["graph_construction_name"],
            "share_vocab": self.use_share_vocab,
            "enc_emb_size": self.opt["model_args"]["graph_initialization_args"]["input_size"],
            "dec_emb_size": self.opt["model_args"]["decoder_args"]["rnn_decoder_share"][
                "input_size"
            ],
            "dynamic_init_graph_name": self.opt["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_name", None),
            "min_word_vocab_freq": self.opt["preprocessing_args"]["min_freq"],
            "pretrained_word_emb_name": self.opt["preprocessing_args"]["pretrained_word_emb_name"],
            "for_inference": 1,
            "reused_vocab_model": self.model.vocab_model,
        }

        # for inference
        inference_dataset = MawpsDatasetForTree(**para_dic)

        self.inference_data_loader = DataLoader(
            inference_dataset.test,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=inference_dataset.collate_fn,
        )

        self.vocab_model = self.model.vocab_model
        self.src_vocab = self.vocab_model.in_word_vocab
        self.tgt_vocab = self.vocab_model.out_word_vocab
        self.share_vocab = self.vocab_model.share_vocab if self.use_share_vocab else None

    def _build_model(self):
        """For encoder-decoder"""
        self.model = Graph2Tree.load_checkpoint(
            self.opt["checkpoint_args"]["out_dir"], self.opt["checkpoint_args"]["checkpoint_name"]
        ).to(self.device)

    def prepare_ext_vocab(self, batch_graph, src_vocab):
        oov_dict = copy.deepcopy(src_vocab)
        token_matrix = []
        for n in batch_graph.node_attributes:
            node_token = n["token"]
            if (n.get("type") is None or n.get("type") == 0) and oov_dict.get_symbol_idx(
                node_token
            ) == oov_dict.get_symbol_idx(oov_dict.unk_token):
                oov_dict.add_symbol(node_token)
            token_matrix.append(oov_dict.get_symbol_idx(node_token))
        batch_graph.node_features["token_id_oov"] = torch.tensor(token_matrix, dtype=torch.long).to(
            self.device
        )
        return oov_dict

    def _build_evaluation(self):
        self.evaluation_metric = SolutionMatch()

    def infer(self):
        self.model.eval()
        pred_list = []
        ground_truth_list = []

        for data in self.inference_data_loader:
            eval_input_graph, _, _ = (
                data["graph_data"],
                data["dec_tree_batch"],
                data["original_dec_tree_batch"],
            )
            eval_input_graph = eval_input_graph.to(self.device)
            oov_dict = self.prepare_ext_vocab(eval_input_graph, self.src_vocab)

            # if self.use_copy:
            #     assert len(batch_original_tree_list) == 1
            #     reference = oov_dict.get_symbol_idx_for_list(batch_original_tree_list[0].split())
            #     eval_vocab = oov_dict
            # else:
            #     assert len(batch_original_tree_list) == 1
            #     reference = self.model.tgt_vocab.get_symbol_idx_for_list(
            #         batch_original_tree_list[0].split()
            #     )
            #     eval_vocab = self.tgt_vocab

            candidate = self.model.translate(
                eval_input_graph,
                oov_dict=oov_dict,
                use_beam_search=True,
                beam_size=self.opt["inference_args"]["beam_size"],
            )

            candidate = [int(c) for c in candidate]
            input_str = " ".join(x["token"] for x in eval_input_graph.node_attributes)
            pred_str = " ".join(self.model.tgt_vocab.get_idx_symbol_for_list(candidate))
            print(input_str)
            print(pred_str + "\n")

            pred_list.append(pred_str)
            ground_truth_list.append(data["original_dec_tree_batch"][0])

        # If the file for inference have meaningful ground truth,
        # we calculate a solution match score for the result.
        if all(DUMMY_STR not in i for i in ground_truth_list):
            print(
                "Solution match score: {:.4f}".format(
                    self.evaluation_metric.calculate_scores(ground_truth_list, pred_list)
                )
            )


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

    start = time.time()
    runner = Mawps(opt=config)
    runner.infer()

    end = time.time()
    print("total time: {} minutes\n".format((end - start) / 60))
