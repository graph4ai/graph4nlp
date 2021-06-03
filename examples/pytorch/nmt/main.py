import logging
import os
from random import shuffle
import resource

from torch.functional import split
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (409600, rlimit[1]))

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ['CUDA_LAUNCH_BLOCKING'] = "5"
from dataset import IWSLT14Dataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import \
    ConstituencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import \
    NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import \
    NodeEmbeddingBasedRefinedGraphConstruction

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from args import get_args
from utils import get_log, wordid2str, WarmupCosineSchedule, save_config
from build_model import get_model
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.evaluation import BLEU



class NMT:
    def __init__(self, opt):
        super(NMT, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        assert self.use_copy == False, print("Copy is not fit to NMT")
        self.use_coverage = self.opt["decoder_args"]["rnn_decoder_share"]["use_coverage"]
        self._build_device(self.opt)
        self._build_logger(self.opt["log_dir"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()
        self._build_loss_function()

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
                              dynamic_init_topology_aux_args=None)


        self.train_dataloader = DataLoader(dataset.train, batch_size=self.opt["batch_size"], shuffle=True,
                                           num_workers=8,
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=self.opt["batch_size"], shuffle=False,
                                           num_workers=8,
                                           collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.opt["batch_size"], shuffle=False, num_workers=8,
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model

    def _build_model(self):

        self.model = get_model(self.opt, vocab_model=self.vocab, device=self.device).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt["learning_rate"])
        self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=self.opt["warmup_steps"], t_total=self.opt["max_steps"])

    def _build_evaluation(self):
        self.metrics = {'BLEU': BLEU(n_grams=[1, 2, 3, 4])}

    def _build_loss_function(self):
        self.loss = Graph2SeqLoss(ignore_index=self.vocab.out_word_vocab.PAD,
                                  use_coverage=self.use_coverage, coverage_weight=0.3)

    def train(self):
        max_score = -1
        self._best_epoch = -1
        self.global_steps = 0
        for epoch in range(200):
            self.model.train()
            self.train_epoch(epoch, split="train")
            # self._adjust_lr(epoch)
            if epoch >= 0:
                score = self.evaluate(epoch=epoch, split="val")
                if score >= max_score:
                    self.logger.info("Best model saved, epoch {}".format(epoch))
                    self.save_checkpoint("best.pth")
                    self._best_epoch = epoch
                max_score = max(max_score, score)
            if epoch >= 30 and self._stop_condition(epoch):
                break
        return max_score

    def _stop_condition(self, epoch, patience=20):
        return epoch > patience + self._best_epoch

    def _adjust_lr(self, epoch):
        def set_lr(optimizer, decay_factor):
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * decay_factor

        epoch_diff = epoch - self.opt["lr_start_decay_epoch"]
        if epoch_diff >= 0 and epoch_diff % self.opt["lr_decay_per_epoch"] == 0:
            if self.opt["learning_rate"] > self.opt["min_lr"]:
                set_lr(self.optimizer, self.opt["lr_decay_rate"])
                self.opt["learning_rate"] = self.opt["learning_rate"] * self.opt["lr_decay_rate"]
                self.logger.info("Learning rate adjusted: {:.5f}".format(self.opt["learning_rate"]))

    def train_epoch(self, epoch, split="train"):
        assert split in ["train"]
        self.logger.info("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_dataloader
        step_all_train = len(dataloader)

        for step, data in enumerate(dataloader):
            graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]
            tgt = tgt.to(self.device)
            graph = graph.to(self.device)
            oov_dict = None
            prob, enc_attn_weights, coverage_vectors = self.model(graph, tgt, oov_dict=oov_dict)
            loss = self.loss(logits=prob, label=tgt, enc_attn_weights=enc_attn_weights,
                             coverage_vectors=coverage_vectors)

            # add graph regularization loss if available
            if graph.graph_attributes.get('graph_reg', None) is not None:
                loss = loss + graph.graph_attributes['graph_reg']

            loss_collect.append(loss.item())
            self.global_steps += 1

            if step % self.opt["loss_display_step"] == 0 and step != 0:
                self.logger.info("Epoch {}: [{} / {}] loss: {:.3f}".format(epoch, step, step_all_train,
                                                                           np.mean(loss_collect)))
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=self.global_steps)
            self.writer.add_scalar("train/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.global_steps)

    def remove_bpe(self, str_with_subword):
        if isinstance(str_with_subword, list):
            return [self.remove_bpe(ss) for ss in str_with_subword]
        symbol = "@@ "
        return str_with_subword.replace(symbol, "").strip()

    def evaluate(self, epoch, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]
            # tgt = tgt.to(self.device)
            graph = graph.to(self.device)

            oov_dict = None
            ref_dict = self.vocab.out_word_vocab

            prob, _, _ = self.model(graph, oov_dict=oov_dict)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), ref_dict)
            pred_collect.extend(self.remove_bpe( pred_str))
            gt_collect.extend(self.remove_bpe(gt_str))

        score = self.metrics["BLEU"].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation results in `{}` split: BLEU-1:{:.4f}\tBLEU-2:{:.4f}\tBLEU-3:{:.4f}\tBLEU-4:{:.4f}".format(split, score[0][0], score[0][1], score[0][2], score[0][3]))
        self.writer.add_scalar(split + "/BLEU@1", score[0][0] * 100, global_step=epoch)
        self.writer.add_scalar(split + "/BLEU@2", score[0][1] * 100, global_step=epoch)
        self.writer.add_scalar(split + "/BLEU@3", score[0][2] * 100, global_step=epoch)
        self.writer.add_scalar(split + "/BLEU@4", score[0][3] * 100, global_step=epoch)
        return score[0][-1]
    
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

    def load_checkpoint(self, checkpoint_name):
        path_dir = os.path.join(self.opt["checkpoint_save_path"], self.opt["name"])
        checkpoint_path = os.path.join(path_dir, checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_name):
        path_dir = os.path.join(self.opt["checkpoint_save_path"], self.opt["name"])
        checkpoint_path = os.path.join(path_dir, checkpoint_name)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    runner = NMT(opt)
    runner.logger.info("------ Running Training ----------")
    runner.logger.info("\tRunner name: {}".format(opt["name"]))
    # save_config(opt, os.path.join(opt["checkpoint_save_path"], opt["name"]))
    max_score = runner.train()
    runner.logger.info("Train finish, best val score: {:.3f}".format(max_score))
    runner.load_checkpoint("best.pth")
    runner.translate()
