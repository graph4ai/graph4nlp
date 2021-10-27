"""
    The advanced inference code.
    In this file, we will run the inference by writing the whole inference pipeline.
    Compared with the inference.py, it is more efficient. It will save the graphs \
        during inference, which support multi-processing when converting the raw inputs to graphs.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab

from args import get_args
from evaluation import ExpressionAccuracy
from utils import get_log, wordid2str


class Jobs:
    def __init__(self, opt):
        super(Jobs, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.opt["decoder_args"]["rnn_decoder_share"]["use_coverage"]
        self._build_device(self.opt)
        self._build_logger(self.opt["log_file"])
        self._build_model()
        self._build_dataloader()
        self._build_evaluation()

    def _build_device(self, opt):
        seed = opt["seed"]
        np.random.seed(seed)
        if opt["use_gpu"] != 0 and torch.cuda.is_available():
            print("[ Using CUDA ]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = torch.device("cuda" if opt["gpu"] < 0 else "cuda:%d" % opt["gpu"])
        else:
            print("[ Using CPU ]")
            device = torch.device("cpu")
        self.device = device

    def _build_logger(self, log_file):
        import os

        log_folder = os.path.split(log_file)[0]
        if not os.path.exists(log_file):
            os.makedirs(log_folder)
        self.logger = get_log(log_file)

    def _build_dataloader(self):
        dataset = JobsDataset(
            root_dir=self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"],
            #   pretrained_word_emb_file=self.opt["pretrained_word_emb_file"],
            pretrained_word_emb_name=self.opt["pretrained_word_emb_name"],
            pretrained_word_emb_url=self.opt["pretrained_word_emb_url"],
            pretrained_word_emb_cache_dir=self.opt["pretrained_word_emb_cache_dir"],
            #   val_split_ratio=self.opt["val_split_ratio"],
            merge_strategy=self.opt["graph_construction_args"]["graph_construction_private"][
                "merge_strategy"
            ],
            edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"][
                "edge_strategy"
            ],
            seed=self.opt["seed"],
            word_emb_size=self.opt["word_emb_size"],
            share_vocab=self.opt["graph_construction_args"]["graph_construction_share"][
                "share_vocab"
            ],
            graph_name=self.opt["graph_construction_args"]["graph_construction_share"][
                "graph_name"
            ],
            dynamic_init_graph_name=self.opt["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_type", None),
            topology_subdir=self.opt["graph_construction_args"]["graph_construction_share"][
                "topology_subdir"
            ],
            thread_number=self.opt["graph_construction_args"]["graph_construction_share"][
                "thread_number"
            ],
            port=self.opt["graph_construction_args"]["graph_construction_share"]["port"],
            for_inference=True,
            reused_vocab_model=self.model.vocab_model,
        )

        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.opt["batch_size"],
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Graph2Seq.load_checkpoint(self.opt["checkpoint_save_path"], "best.pt").to(
            self.device
        )

    def _build_evaluation(self):
        self.metrics = [ExpressionAccuracy()]

    def evaluate(self, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.test_dataloader
        for data in dataloader:
            graph, gt_str = data["graph_data"], data["output_str"]
            graph = graph.to(self.device)
            if self.use_copy:
                oov_dict = prepare_ext_vocab(
                    batch_graph=graph, vocab=self.vocab, device=self.device
                )
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            prob, _, _ = self.model(graph, oov_dict=oov_dict)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), ref_dict)
            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format(split, score))
        return score

    @torch.no_grad()
    def translate(self):
        self.model.eval()

        pred_collect = []
        gt_collect = []
        dataloader = self.test_dataloader
        for data in dataloader:
            graph, gt_str = data["graph_data"], data["output_str"]
            graph = graph.to(self.device)
            if self.use_copy:
                oov_dict = prepare_ext_vocab(
                    batch_graph=graph, vocab=self.vocab, device=self.device
                )
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            pred = self.model.translate(batch_graph=graph, oov_dict=oov_dict, beam_size=4, topk=1)

            pred_ids = pred[:, 0, :]  # we just use the top-1

            pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format("test", score))
        return score


if __name__ == "__main__":
    opt = get_args()
    runner = Jobs(opt)
    runner.translate()
