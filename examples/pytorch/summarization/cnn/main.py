import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "3"

from .dataset import CNNDataset
# from .model_onmt import Graph2seq
from .model import Graph2seq
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
# from graph4nlp.pytorch.modules.graph_construction.linear_graph_construction import LinearGraphConstruction
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from graph4nlp.pytorch.data.dataset import CNNSeq2SeqDataset
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
import copy
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import BeamSearchStrategy

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from .config import get_args
from .utils import get_log, wordid2str
from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
import time
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab


class CNN:
    def __init__(self, opt):
        super(CNN, self).__init__()
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
        # if self.opt.topology_subdir == 'ie':
        #     graph_type = 'static'
        #     topology_builder = IEBasedGraphConstruction
        #     topology_subdir = 'IEGraph'
        #     dynamic_graph_type = None
        #     dynamic_init_topology_builder = None
        # # elif self.opt.topology_subdir == 'DependencyGraph3':
        # elif 'DependencyGraph' in self.opt.topology_subdir:
        #     graph_type = 'static'
        #     topology_builder = DependencyBasedGraphConstruction
        #     topology_subdir = self.opt.topology_subdir
        #     dynamic_graph_type = None
        #     dynamic_init_topology_builder = None
        # elif self.opt.topology_subdir == 'node_emb':
        #     graph_type = 'dynamic'
        #     topology_builder = NodeEmbeddingBasedGraphConstruction
        #     topology_subdir = 'NodeEmb'
        #     dynamic_graph_type = 'node_emb'
        #     dynamic_init_topology_builder = DependencyBasedGraphConstruction
        # elif self.opt.topology_subdir == 'LinearGraph':
        #     graph_type = 'static'
        #     topology_builder = LinearGraphConstruction
        #     topology_subdir = self.opt.topology_subdir
        #     dynamic_graph_type = None
        #     dynamic_init_topology_builder = None
        # else:
        #     raise NotImplementedError()

        # dataset = CNNDataset(root_dir=self.opt.root_dir,
        #                      graph_type=graph_type,
        #                      topology_builder=topology_builder,
        #                      topology_subdir=topology_subdir,
        #                      dynamic_graph_type=dynamic_graph_type,
        #                      dynamic_init_topology_builder=dynamic_init_topology_builder,
        #                      dynamic_init_topology_aux_args={'dummy_param': 0})

        dataset = CNNSeq2SeqDataset(root_dir=self.opt.root_dir,
                                    word_emb_size=self.opt.word_emb_size,
                                    topology_builder=DependencyBasedGraphConstruction,
                                    topology_subdir=self.opt.topology_subdir, share_vocab=True)

        self.train_dataloader = DataLoader(dataset.train, batch_size=self.opt.batch_size, shuffle=True, num_workers=0,
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=self.opt.batch_size, shuffle=False, num_workers=0,
                                         collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.opt.batch_size, shuffle=False, num_workers=0,
                                          collate_fn=dataset.collate_fn)
        self.vocab: VocabModel = dataset.vocab_model

    def _build_model(self):
        self.model = Graph2seq(self.vocab,
                               use_copy=self.opt.use_copy,
                               use_coverage=self.opt.use_coverage,
                               gnn=self.opt.gnn,
                               device=self.device,
                               rnn_dropout=self.opt.rnn_dropout,
                               word_dropout=self.opt.word_dropout,
                               hidden_size=self.opt.hidden_size,
                               word_emb_size=self.opt.word_emb_size).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt.learning_rate)

    def _build_evaluation(self):
        self.metrics = [ROUGE()]

    def train(self):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(200):
            # score = self.evaluate(split="val")
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
                # score = self.evaluate(split="test")
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

    def prepare_ext_vocab_seq2seq(self, src_seq, gt_src, vocab, gt_str=None):
        oov_dict = copy.deepcopy(vocab.in_word_vocab)
        for i in range(src_seq.shape[0]):

            seq = src_seq[i]
            src_str = gt_src[i]
            if oov_dict.tokenizer is None:
                tokenizerd = src_str.split()
            else:
                tokenizerd = oov_dict.tokenizer(src_str)
            for j in range(len(tokenizerd)):
                if oov_dict.getIndex(tokenizerd[j]) == oov_dict.UNK:
                    oov_dict._add_words([tokenizerd[j]])
                if seq[j].item() == oov_dict.UNK:
                    seq[j] = oov_dict.getIndex(tokenizerd[j])

        if gt_str is not None:
            oov_tgt_collect = []
            for s in gt_str:
                oov_tgt = oov_dict.to_index_sequence(s)
                oov_tgt.append(oov_dict.EOS)
                oov_tgt = np.array(oov_tgt)
                oov_tgt_collect.append(oov_tgt)

            output_pad = pad_2d_vals_no_size(oov_tgt_collect)

            tgt_seq = torch.from_numpy(output_pad).long().to(self.device)
            return oov_dict, tgt_seq
        else:
            return oov_dict

    def train_epoch(self, epoch, split="train"):
        assert split in ["train"]
        self.logger.info("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_dataloader
        step_all_train = len(dataloader)
        start = time.time()
        for step, data in enumerate(dataloader):
            # graph_list, tgt = data
            src_seq, src_len, tgt_seq, src_str, tgt_str = data
            src_seq = src_seq.to(self.device)
            src_len = src_len.to(self.device)
            tgt = tgt_seq.to(self.device)
            oov_dict = None
            if self.opt.use_copy:
                # oov_dict = copy.deepcopy(self.vocab.in_word_vocab)
                oov_dict, tgt = self.prepare_ext_vocab_seq2seq(src_seq, src_str, self.vocab, tgt_str)

            _, loss = self.model(src_seq, src_len, tgt, require_loss=True, oov_dict=oov_dict)
            # _, loss = self.model(graph_list, tgt, require_loss=True)
            loss_collect.append(loss.item())
            if step % self.opt.loss_display_step == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "Epoch {}: [{} / {}] loss: {:.3f}, time cost: {:.3f}".format(epoch, step, step_all_train,
                                                                                 np.mean(loss_collect), end - start))
                start = time.time()
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, split="val", test_mode=False):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            # graph_list, tgt = data
            src_seq, src_len, tgt_seq, src_str, tgt_str = data
            src_seq = src_seq.to(self.device)
            src_len = src_len.to(self.device)
            tgt = tgt_seq.to(self.device)
            oov_dict = None
            if self.opt.use_copy:
                # oov_dict = copy.deepcopy(self.vocab.in_word_vocab)
                oov_dict, tgt = self.prepare_ext_vocab_seq2seq(src_seq, src_str, self.vocab, tgt_str)
            # prob = self.model(src_seq, src_len, tgt, require_loss=True)
            prob = self.model(src_seq, src_len, require_loss=False, oov_dict=oov_dict)
            # prob = self.model(graph_list, require_loss=False)
            pred = prob.argmax(dim=-1)
            # pred = prob[0].argmax(dim=-1).reshape(prob[0].size()[1], prob[0].size()[0])

            pred_str = wordid2str(pred.detach().cpu(), self.vocab.out_word_vocab)
            tgt_str = wordid2str(tgt, self.vocab.out_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        if test_mode==True:
            with open('cnn_pred_output.txt','w+') as f:
                for line in pred_collect:
                    f.write(line+'\n')

            with open('cnn_tgt_output.txt','w+') as f:
                for line in gt_collect:
                    f.write(line+'\n')

        score, _ = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation results in `{}` split".format(split))
        self.logger.info("ROUGE: {:.3f}".format(score))
        return score

    def translate(self):
        self.model.eval()
        generator = BeamSearchStrategy(beam_size=self.opt.beam_size, vocab=self.model.seq_decoder.vocab, rnn_type="LSTM",
                                       decoder=self.model.seq_decoder, use_copy=self.opt.use_copy, use_coverage=self.opt.use_copy)

        pred_collect = []
        gt_collect = []
        dataloader = self.test_dataloader
        for data in dataloader:
            src_seq, src_len, tgt_seq, src_str, tgt_str = data
            src_seq = src_seq.to(self.device)
            src_len = src_len.to(self.device)
            tgt = tgt_seq.to(self.device)
            oov_dict = None
            if self.opt.use_copy:
                oov_dict = self.prepare_ext_vocab_seq2seq(src_seq, src_str, self.vocab, tgt_str)
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            src_seq = src_seq.transpose(0, 1)
            enc_state, memory_bank, lengths = self.model.seq_encoder(src_seq.unsqueeze(-1), src_len)

            # down-task
            max_len = max(src_len)
            enc_mask = torch.arange(max_len).expand(len(src_len), max_len) < src_len.cpu().unsqueeze(1)
            enc_mask = (enc_mask.long() - 1).to(self.device)
            prob = generator._beam_search(memory_bank.transpose(0, 1),
                                          graph_node_mask=enc_mask,
                                          src_seq=src_seq.transpose(0, 1),
                                          oov_dict=oov_dict,
                                          graph_level_embedding=torch.cat(
                                              (enc_state[0][0],
                                               enc_state[0][1]), dim=1))

            pred_ids = torch.zeros(len(prob), self.opt.decoder_length).fill_(Vocab.EOS).to(tgt.device).int()
            for i, item in enumerate(prob):
                item = item[0]
                seq = [i.view(1, 1) for i in item]
                seq = torch.cat(seq, dim=1)
                pred_ids[i, :seq.shape[1]] = seq

            pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)
            tgt_str = wordid2str(tgt, self.vocab.in_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        score, _ = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format("test", score))
        return score

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt.checkpoint_save_path, checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    runner = CNN(opt)
    # max_score = runner.train()
    # print("Train finish, best val score: {:.3f}".format(max_score))
    runner.load_checkpoint('best.pth')
    runner.evaluate(split='test', test_mode=True)
    # runner.translate()
