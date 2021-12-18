import os
import resource
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.evaluation import BLEU

from args import get_args
from build_model import get_model
from dataset import IWSLT14Dataset
from utils import WarmupCosineSchedule, get_log, wordid2str

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)


class NMT:
    def __init__(self, opt):
        super(NMT, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        assert self.use_copy is False, print("Copy is not fit to NMT")
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
        dataset = IWSLT14Dataset(
            root_dir=self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"],
            val_split_ratio=self.opt["val_split_ratio"],
            merge_strategy=self.opt["graph_construction_args"]["graph_construction_private"][
                "merge_strategy"
            ],
            edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"][
                "edge_strategy"
            ],
            seed=self.opt["seed"],
            word_emb_size=self.opt["word_emb_size"],
            share_vocab=self.opt["share_vocab"],
            graph_name=self.opt["graph_construction_args"]["graph_construction_share"][
                "graph_name"
            ],
            topology_subdir=self.opt["graph_construction_args"]["graph_construction_share"][
                "topology_subdir"
            ],
        )

        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=self.opt["batch_size"],
            shuffle=True,
            num_workers=8,
            collate_fn=dataset.collate_fn,
        )
        self.val_dataloader = DataLoader(
            dataset.val,
            batch_size=self.opt["batch_size"],
            shuffle=False,
            num_workers=8,
            collate_fn=dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.opt["batch_size"],
            shuffle=False,
            num_workers=8,
            collate_fn=dataset.collate_fn,
        )
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = get_model(self.opt, vocab_model=self.vocab, device=self.device).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt["learning_rate"])
        self.scheduler = WarmupCosineSchedule(
            self.optimizer, warmup_steps=self.opt["warmup_steps"], t_total=self.opt["max_steps"]
        )

    def _build_evaluation(self):
        self.metrics = {"BLEU": BLEU(n_grams=[1, 2, 3, 4])}

    def _build_loss_function(self):
        self.loss = Graph2SeqLoss(
            ignore_index=self.vocab.out_word_vocab.PAD,
            use_coverage=self.use_coverage,
            coverage_weight=0.3,
        )

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
                    self.model.save_checkpoint(
                        os.path.join("examples/pytorch/nmt/save", opt["name"]), "best.pth"
                    )
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
                group["lr"] = group["lr"] * decay_factor

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
            graph, tgt = data["graph_data"], data["tgt_seq"]
            tgt = tgt.to(self.device)
            graph = graph.to(self.device)
            oov_dict = None
            prob, enc_attn_weights, coverage_vectors = self.model(graph, tgt, oov_dict=oov_dict)
            loss = self.loss(
                logits=prob,
                label=tgt,
                enc_attn_weights=enc_attn_weights,
                coverage_vectors=coverage_vectors,
            )

            # add graph regularization loss if available
            if graph.graph_attributes.get("graph_reg", None) is not None:
                loss = loss + graph.graph_attributes["graph_reg"]

            loss_collect.append(loss.item())
            self.global_steps += 1

            if step % self.opt["loss_display_step"] == 0 and step != 0:
                self.logger.info(
                    "Epoch {}: [{} / {}] loss: {:.3f}".format(
                        epoch, step, step_all_train, np.mean(loss_collect)
                    )
                )
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalar(
                "train/loss", scalar_value=loss.item(), global_step=self.global_steps
            )
            self.writer.add_scalar(
                "train/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.global_steps
            )

    def remove_bpe(self, str_with_subword):
        if isinstance(str_with_subword, list):
            return [self.remove_bpe(ss) for ss in str_with_subword]
        symbol = "@@ "
        return str_with_subword.replace(symbol, "").strip()

    @torch.no_grad()
    def evaluate(self, epoch, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            graph, gt_str = data["graph_data"], data["output_str"]
            graph = graph.to(self.device)

            oov_dict = None
            ref_dict = self.vocab.out_word_vocab

            prob, _, _ = self.model(graph, oov_dict=oov_dict)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), ref_dict)
            pred_collect.extend(self.remove_bpe(pred_str))
            gt_collect.extend(self.remove_bpe(gt_str))

        score = self.metrics["BLEU"].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info(
            "Evaluation results in `{}` split: BLEU-1:{:.4f}\tBLEU-2:{:.4f}\tBLEU-3:{:.4f}\t"
            "BLEU-4:{:.4f}".format(split, score[0][0], score[0][1], score[0][2], score[0][3])
        )
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
    runner = NMT(opt)
    runner.logger.info("------ Running Training ----------")
    runner.logger.info("\tRunner name: {}".format(opt["name"]))
    # save_config(opt, os.path.join(opt["checkpoint_save_path"], opt["name"]))
    max_score = runner.train()
    runner.logger.info("Train finish, best val score: {:.3f}".format(max_score))
    runner.model = Graph2Seq.load_checkpoint(
        os.path.join("examples/pytorch/nmt/save", opt["name"]), "best.pth"
    ).to(runner.device)
    runner.translate()
