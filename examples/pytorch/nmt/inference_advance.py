import os
import resource
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.evaluation import BLEU
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config

from args import get_args
from dataset import IWSLT14Dataset
from utils import get_log, wordid2str

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)


class NMT:
    def __init__(self, opt):
        super(NMT, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["model_args"]["decoder_args"]["rnn_decoder_share"]["use_copy"]
        assert self.use_copy is False, print("Copy is not fit to NMT")
        self.use_coverage = self.opt["model_args"]["decoder_args"]["rnn_decoder_share"][
            "use\
            _coverage"
        ]
        self._build_device(self.opt)
        self._build_logger(self.opt["training_args"]["log_dir"])
        self._build_model()
        self._build_dataloader()
        self._build_evaluation()

    def _build_device(self, opt):
        seed = opt["env_args"]["seed"]
        np.random.seed(seed)
        if opt["env_args"]["use_gpu"] != 0 and torch.cuda.is_available():
            print("[ Using CUDA ]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = torch.device(
                "cuda"
                if opt["env_args"]["gpuid"] < 0
                else "cuda:%d"
                % opt[
                    "\
                env_args"
                ]["gpuid"]
            )
        else:
            print("[ Using CPU ]")
            device = torch.device("cpu")
        self.device = device

    def _build_logger(self, log_dir):
        log_path = os.path.join(log_dir, self.opt["training_args"]["name"])
        logger_path = os.path.join(log_path, "txt")
        tensorboard_path = os.path.join(log_path, "tensorboard")
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        self.logger = get_log(logger_path + "log.txt")
        self.writer = SummaryWriter(log_dir=tensorboard_path)

    def _build_dataloader(self):
        dataset = IWSLT14Dataset(
            root_dir=self.opt["model_args"]["graph_construction_args"][
                "\
                graph_construction_share"
            ]["root_dir"],
            val_split_ratio=self.opt["preprocessing_args"]["val_split_ratio"],
            merge_strategy=self.opt["model_args"]["graph_construction_args"][
                "\
                graph_construction_private"
            ]["merge_strategy"],
            edge_strategy=self.opt["model_args"]["graph_construction_args"][
                "\
                graph_construction_private"
            ]["edge_strategy"],
            seed=self.opt["env_args"]["seed"],
            word_emb_size=self.opt["preprocessing_args"]["word_emb_size"],
            share_vocab=self.opt["model_args"]["graph_construction_args"][
                "\
                graph_construction_share"
            ]["share_vocab"],
            graph_construction_name=self.opt["model_args"]["graph_construction_name"],
            topology_subdir=self.opt["model_args"]["graph_construction_args"][
                "\
                graph_construction_share"
            ]["topology_subdir"],
            for_inference=True,
            reused_vocab_model=self.model.vocab_model,
        )

        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.opt["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.opt["training_args"]["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Graph2Seq.load_checkpoint(
            os.path.join(
                self.opt["checkpoint_args"]["checkpoint_save_path"],
                self.opt[
                    "\
                training_args"
                ]["name"],
            ),
            "best.pth",
        ).to(self.device)

    def _build_evaluation(self):
        self.metrics = {"BLEU": BLEU(n_grams=[1, 2, 3, 4])}

    def remove_bpe(self, str_with_subword):
        if isinstance(str_with_subword, list):
            return [self.remove_bpe(ss) for ss in str_with_subword]
        symbol = "@@ "
        return str_with_subword.replace(symbol, "").strip()

    @torch.no_grad()
    def translate(self):
        self.model.eval()

        pred_collect = []
        gt_collect = []
        dataloader = self.test_dataloader
        for data in dataloader:
            batch_graph, gt_str = data["graph_data"], data["output_str"]

            oov_dict = None
            ref_dict = self.vocab.out_word_vocab
            batch_graph = batch_graph.to(self.device)

            pred = self.model.translate(
                batch_graph=batch_graph, oov_dict=oov_dict, beam_size=3, topk=1
            )

            pred_ids = pred[:, 0, :]  # we just use the top-1

            pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics["BLEU"].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info(
            "Evaluation results in `{}` split: BLEU-1:{:.4f}\tBLEU-2:{:.4f}\tBLEU-3:{:.4f}\t"
            "BLEU-4:{:.4f}".format("test", score[0][0], score[0][1], score[0][2], score[0][3])
        )
        self.writer.add_scalar("test" + "/BLEU@1", score[0][0] * 100, global_step=0)
        self.writer.add_scalar("test" + "/BLEU@2", score[0][1] * 100, global_step=0)
        self.writer.add_scalar("test" + "/BLEU@3", score[0][2] * 100, global_step=0)
        self.writer.add_scalar("test" + "/BLEU@4", score[0][3] * 100, global_step=0)
        return score


if __name__ == "__main__":
    opt = get_args()
    config = load_json_config(opt["json_config"])
    runner = NMT(config)
    runner.logger.info("------ Running Inference ----------")
    runner.logger.info("\tRunner name: {}".format(config["training_args"]["name"]))
    runner.translate()
