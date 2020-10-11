import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ['CUDA_LAUNCH_BLOCKING'] = "5"
from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import NodeEmbeddingBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import NodeEmbeddingBasedRefinedGraphConstruction

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from .args import get_args
from .evaluation import ExpressionAccuracy
from .utils import get_log, wordid2str
from .model import Graph2seq
from graph4nlp.pytorch.data.data import from_batch, GraphData
import copy
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size


class Jobs:
    def __init__(self, opt):
        super(Jobs, self).__init__()
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
        if self.opt.graph_type == "dependency":
            topology_builder = DependencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt.graph_type == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt.graph_type == "node_emb":
            topology_builder = NodeEmbeddingBasedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_topology_builder = None
        elif self.opt.graph_type == "node_emb_refined":
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            graph_type = 'dynamic'
            if self.opt.dynamic_init_graph_type is None or self.opt.dynamic_init_graph_type == 'line':
                dynamic_init_topology_builder = None
            elif self.opt.dynamic_init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif self.opt.dynamic_init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            else:
                # dynamic_init_topology_builder
                raise RuntimeError('Define your own dynamic_init_topology_builder')
        else:
            raise NotImplementedError("Define your topology builder.")

        dataset = JobsDataset.from_args(args=self.opt, topology_builder=topology_builder,
                                        graph_type=graph_type,
                                        dynamic_graph_type=graph_type if graph_type in ('node_emb', 'node_emb_refined') else None,
                                        dynamic_init_topology_builder=dynamic_init_topology_builder)

        self.train_dataloader = DataLoader(dataset.train, batch_size=24, shuffle=True, num_workers=1,
                                           collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=24, shuffle=False, num_workers=1,
                                          collate_fn=dataset.collate_fn)

        self.vocab = dataset.vocab_model
        # from examples.pytorch.semantic_parsing.graph2seq.utils import get_glove_weights
        # pretrained_weight = get_glove_weights(self.vocab.in_word_vocab)
        # self.vocab.in_word_vocab.embeddings = pretrained_weight.numpy()

    def _build_model(self):
        self.model = Graph2seq.from_args(vocab=self.vocab, args=self.opt, device=self.device).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt.learning_rate)

    def _build_evaluation(self):
        self.metrics = [ExpressionAccuracy()]

    def train(self):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(200):
            self.model.train()
            self.train_epoch(epoch, split="train")
            self._adjust_lr(epoch)
            if epoch >= 0:
                score = self.evaluate(split="test")
                if score >= max_score:
                    self.logger.info("Best model saved, epoch {}".format(epoch))
                    self.save_checkpoint("best.pth")
                    self._best_epoch = epoch
                max_score = max(max_score, score)
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

    def prepare_ext_vocab(self, batch, vocab, gt_str=None):
        oov_dict = copy.deepcopy(vocab.in_word_vocab)
        for g in batch:
            token_matrix = []
            for node_idx in range(g.get_node_num()):
                node_token = g.node_attributes[node_idx]['token']
                if oov_dict.getIndex(node_token) == oov_dict.UNK:
                    oov_dict._add_words(node_token)
                token_matrix.append([oov_dict.getIndex(node_token)])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long).to(self.device)
            g.node_features['token_id_oov'] = token_matrix

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
        for step, data in enumerate(dataloader):
            graph_list, tgt, gt_str = data
            tgt = tgt.to(self.device)
            oov_dict, tgt_oov = self.prepare_ext_vocab(graph_list, self.vocab, gt_str=gt_str)

            _, loss = self.model(graph_list, tgt_oov, oov_dict=oov_dict, require_loss=True)
            loss_collect.append(loss.item())
            if step % self.opt.loss_display_step == 0 and step != 0:
                self.logger.info("Epoch {}: [{} / {}] loss: {:.3f}".format(epoch, step, step_all_train,
                                                                           np.mean(loss_collect)))
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            graph_list, tgt, gt_str = data
            oov_dict = self.prepare_ext_vocab(graph_list, self.vocab)
            prob = self.model(graph_list, oov_dict=oov_dict, require_loss=False)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), oov_dict)
            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format(split, score))
        return score

    def translate(self):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        dataloader = self.test_dataloader
        for data in dataloader:
            graph_list, tgt = data
            batch_graph = self.model.graph_topology(graph_list)

            # run GNN
            batch_graph = self.model.gnn_encoder(batch_graph)
            batch_graph.node_features["rnn_emb"] = batch_graph.node_features['node_feat']

            # down-task
            params = self.model.seq_decoder._extract_params(from_batch(batch_graph))
            params['tgt_seq'] = None
            params['src_seq'] = None
            params['beam_width'] = 5
            params['topk'] = 1
            params['teacher_forcing_rate'] = 0

            prob = self.model.seq_decoder.beam_search(**params)
            pred_ids = torch.zeros(len(prob), 50).fill_(2).to(tgt.device).int()
            seq_collect = []
            for i, item in enumerate(prob):
                item = item[0]
                seq = [i.view(1, 1) for i in item]
                seq = torch.cat(seq, dim=1)
                pred_ids[i, :seq.shape[1]] = seq

            pred_str = wordid2str(pred_ids.detach().cpu(), self.vocab.in_word_vocab)
            tgt_str = wordid2str(tgt, self.vocab.in_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
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
    runner = Jobs(opt)
    max_score = runner.train()
    print("Train finish, best val score: {:.3f}".format(max_score))
    runner.load_checkpoint("best.pth")
    runner.evaluate(split="test")
    # runner.translate()
