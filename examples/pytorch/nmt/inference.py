import logging
import os
import resource
import sys
from random import shuffle
import numpy as np
import torch
import torch.optim as optim
from torch.functional import split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.evaluation import BLEU
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import (
    ConstituencyBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import (
    DependencyBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import (
    NodeEmbeddingBasedRefinedGraphConstruction,
)
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab

from args import get_args
from build_model import get_model
# os.environ['CUDA_LAUNCH_BLOCKING'] = "5"
from dataset import IWSLT14Dataset
from utils import WarmupCosineSchedule, get_log, save_config, wordid2str

sys.path.append("/export/shenkai/shiina/lib/graph4nlp")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (409600, rlimit[1]))

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"







class NMT:
    def __init__(self, opt):
        super(NMT, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        assert self.use_copy == False, print("Copy is not fit to NMT")
        self.use_coverage = self.opt["decoder_args"]["rnn_decoder_share"]["use_coverage"]
        self._build_device(self.opt)
        self._build_logger(self.opt["log_dir"])
        self._build_model()
        self._build_dataloader()
        self._build_evaluation()

    def _build_device(self, opt):
        seed = opt["seed"]
        np.random.seed(seed)
        if opt["use_gpu"] != 0 and torch.cuda.is_available():
            print('[ Using CUDA ]')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn
            cudnn.benchmark = True
            device = torch.device('cuda' if opt["gpu"] < 0 else 'cuda:%d' % opt["gpu"])
        else:
            print('[ Using CPU ]')
            device = torch.device('cpu')
        self.device = device

    def _build_logger(self, log_dir):
        log_path = os.path.join(log_dir, self.opt["name"])
        logger_path = os.path.join(log_path, "txt")
        tensorboard_path = os.path.join(log_path, "tensorboard")
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        self.logger = get_log(logger_path + "log.txt")
        self.writer = SummaryWriter(log_dir=tensorboard_path)

    def _build_dataloader(self):
        if self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "dependency":
            topology_builder = DependencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "node_emb":
            topology_builder = NodeEmbeddingBasedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "node_emb_refined":
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_graph_type = self.opt["graph_construction_args"]["graph_construction_private"][
                "dynamic_init_graph_type"]
            if dynamic_init_graph_type is None or dynamic_init_graph_type == 'line':
                dynamic_init_topology_builder = None
            elif dynamic_init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif dynamic_init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            else:
                # dynamic_init_topology_builder
                raise RuntimeError('Define your own dynamic_init_topology_builder')
        else:
            raise NotImplementedError("Define your topology builder.")

        dataset = IWSLT14Dataset(root_dir=self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"],
                              val_split_ratio=self.opt["val_split_ratio"],
                              merge_strategy=self.opt["graph_construction_args"]["graph_construction_private"][
                                  "merge_strategy"],
                              edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"][
                                  "edge_strategy"],
                              seed=self.opt["seed"],
                              word_emb_size=self.opt["word_emb_size"], share_vocab=self.opt["share_vocab"],
                              graph_type=graph_type,
                              topology_builder=topology_builder,
                              topology_subdir=self.opt["graph_construction_args"]["graph_construction_share"][
                                  "topology_subdir"],
                              dynamic_graph_type=self.opt["graph_construction_args"]["graph_construction_share"][
                                  "graph_type"],
                              dynamic_init_topology_builder=dynamic_init_topology_builder,
                              dynamic_init_topology_aux_args=None,
                              for_inference=True,
                              reused_vocab_model=self.model.vocab_model)

        self.test_dataloader = DataLoader(dataset.test, batch_size=self.opt["batch_size"], shuffle=False, num_workers=8,
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Graph2Seq.load_checkpoint(os.path.join("examples/pytorch/nmt/save", opt["name"]), "best.pth").to(self.device)

    def _build_evaluation(self):
        self.metrics = {'BLEU': BLEU(n_grams=[1, 2, 3, 4])}

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
            batch_graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]

            oov_dict = None
            ref_dict = self.vocab.out_word_vocab
            batch_graph = batch_graph.to(self.device)

            pred = self.model.translate(batch_graph=batch_graph, oov_dict=oov_dict, beam_size=3, topk=1)

            pred_ids = pred[:, 0, :]  # we just use the top-1

            pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics["BLEU"].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation results in `{}` split: BLEU-1:{:.4f}\tBLEU-2:{:.4f}\tBLEU-3:{:.4f}\tBLEU-4:{:.4f}".format("test", score[0][0], score[0][1], score[0][2], score[0][3]))
        self.writer.add_scalar("test" + "/BLEU@1", score[0][0] * 100, global_step=0)
        self.writer.add_scalar("test" + "/BLEU@2", score[0][1] * 100, global_step=0)
        self.writer.add_scalar("test" + "/BLEU@3", score[0][2] * 100, global_step=0)
        self.writer.add_scalar("test" + "/BLEU@4", score[0][3] * 100, global_step=0)
        return score


if __name__ == "__main__":
    opt = get_args()
    runner = NMT(opt)
    runner.logger.info("------ Running Training ----------")
    runner.logger.info("\tRunner name: {}".format(opt["name"]))
    runner.translate()
