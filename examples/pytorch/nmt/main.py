# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from examples.pytorch.nmt.dataset import EuroparlNMTDataset
from examples.pytorch.nmt.model import Graph2seq
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from .config import get_args
from .utils import get_log, wordid2str
from graph4nlp.pytorch.modules.evaluation.bleu import BLEU
import time

class NMT:
    def __init__(self, opt):
        super(NMT, self).__init__()
        self.opt = opt
        self._build_device(self.opt)
        self._build_logger(self.opt.log_file)
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_device(self, opt):
        seed = opt.seed
        np.random.seed(seed)
        if opt.use_gpu != 0 and torch.cuda.is_available():
            print('[ Using CUDA ]')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn
            cudnn.benchmark = True
            device = torch.device('cuda' if opt.gpu < 0 else 'cuda:%d' % opt.gpu)
        else:
            print('[ Using CPU ]')
            device = torch.device('cpu')
        self.device = device

    def _build_logger(self, log_file):
        self.logger = get_log(log_file)

    def _build_dataloader(self):

        dataset = EuroparlNMTDataset(root_dir="/home/shiina/shiina/lib/dataset/news-commentary-v11/de-en",
                                     topology_builder=DependencyBasedGraphConstruction,
                                     topology_subdir='DependencyGraph', share_vocab=False)

        self.train_dataloader = DataLoader(dataset.train, batch_size=30, shuffle=True, num_workers=10,
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=30, shuffle=False, num_workers=10,
                                           collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=30, shuffle=False, num_workers=10,
                                          collate_fn=dataset.collate_fn)
        self.vocab: VocabModel = dataset.vocab_model
        import torchtext.vocab as vocab
        glove = vocab.GloVe
        glove.url["de"] = "/home/shiina/shiina/lib/graph4nlp/.vector_cache/glove.de.300d.txt"
        from .utils import get_glove_weights
        en = glove(name='6B', dim=300)

        pretrained_weight = get_glove_weights(en, self.vocab.in_word_vocab)
        self.vocab.in_word_vocab.embeddings = pretrained_weight.numpy()
        print("English word embedding loaded")

        de = glove(name='de', dim=300)

        pretrained_weight = get_glove_weights(de, self.vocab.out_word_vocab)
        self.vocab.out_word_vocab.embeddings = pretrained_weight.numpy()
        print("De word embedding loaded")



    def _build_model(self):
        self.model = Graph2seq(self.vocab, device=self.device).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt.learning_rate)

    def _build_evaluation(self):
        self.metrics = [BLEU([1, 2, 3, 4])]

    def train(self):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(200):
            self.model.train()
            self.train_epoch(epoch, split="train")
            self._adjust_lr(epoch)
            if epoch >= 0:
                score = self.evaluate(split="val")
                if score >= max_score:
                    self.logger.info("Best model saved, epoch {}".format(epoch))
                    self.save_checkpoint("best.pth")
                    self._best_epoch = epoch
                max_score = max(max_score, score)
                score = self.evaluate(split="test")
            if epoch >= 30 and self._stop_condition(epoch):
                break
        return max_score

    def _stop_condition(self, epoch, patience=2000):
        return epoch > patience + self._best_epoch

    def _adjust_lr(self, epoch):
        def set_lr(optimizer, decay_factor):
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * decay_factor

        epoch_diff = epoch - self.opt.lr_start_decay_epoch
        if epoch_diff >= 0 and epoch_diff % self.opt.lr_decay_per_epoch == 0:
            if self.opt.learning_rate > self.opt.min_lr:
                set_lr(self.optimizer, self.opt.lr_decay_rate)
                self.opt.learning_rate = self.opt.learning_rate * self.opt.lr_decay_rate
                self.logger.info("Learning rate adjusted: {:.5f}".format(self.opt.learning_rate))

    def train_epoch(self, epoch, split="train"):
        assert split in ["train"]
        self.logger.info("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_dataloader
        step_all_train = len(dataloader)
        start = time.time()
        for step, data in enumerate(dataloader):
            graph_list, tgt = data
            tgt = tgt.to(self.device)
            _, loss = self.model(graph_list, tgt, require_loss=True)
            loss_collect.append(loss.item())
            if step % self.opt.loss_display_step == 0 and step != 0:
                end = time.time()
                self.logger.info("Epoch {}: [{} / {}] loss: {:.3f}, time cost: {:.3f}".format(epoch, step, step_all_train,
                                                                           np.mean(loss_collect), end-start))
                start = time.time()
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            graph_list, tgt = data
            prob = self.model(graph_list, require_loss=False)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), self.vocab.out_word_vocab)
            tgt_str = wordid2str(tgt, self.vocab.out_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        score, _ = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation results in `{}` split".format(split))
        self.logger.info("BLEU @1: {:.3f}, @2: {:.3f}, @3: {:.3f}, @4: {:.4f}".format(score[0], score[1],
                                                                                      score[2], score[3]))
        return score[3]

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    runner = NMT(opt)
    max_score = runner.train()
    print("Train finish, best val score: {:.3f}".format(max_score))
    runner.load_checkpoint("best_all.pth")
    runner.evaluate(split="test")