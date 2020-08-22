import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_embedding.gat import GAT
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab


class ExpressionAccuracy(EvaluationMetricBase):
    def __init__(self):
        super(ExpressionAccuracy, self).__init__()

    def calculate_scores(self, ground_truth, predict):
        correct = 0
        assert len(ground_truth) == len(predict)
        for gt, pred in zip(ground_truth, predict):
            print("ground truth: ", gt)
            print("prediction: ", pred)

            if gt == pred:
                correct += 1.
        return correct / len(ground_truth)


def logits2seq(prob: torch.Tensor):
    ids = prob.argmax(dim=-1)
    return ids


def wordid2str(word_ids, vocab: Vocab):
    ret = []
    assert len(word_ids.shape) == 2, print(word_ids.shape)
    for i in range(word_ids.shape[0]):
        id_list = word_ids[i, :]
        ret_inst = []
        for j in range(id_list.shape[0]):
            if id_list[j] == vocab.EOS:
                break
            token = vocab.getWord(id_list[j])
            ret_inst.append(token)
        ret.append(" ".join(ret_inst))
    return ret


class Graph2seqLoss(nn.Module):
    def __init__(self, vocab: Vocab):
        super(Graph2seqLoss, self).__init__()
        self.loss_func = nn.NLLLoss()
        self.VERY_SMALL_NUMBER = 1e-31
        self.vocab = vocab

    def forward(self, prob, gt):
        assert prob.shape[0:1] == gt.shape[0:1]
        assert len(prob.shape) == 3
        log_prob = torch.log(prob + self.VERY_SMALL_NUMBER)
        batch_size = gt.shape[0]
        step = gt.shape[1]

        mask = 1 - gt.data.eq(self.vocab.PAD).float()

        prob_select = torch.gather(log_prob.view(batch_size * step, -1), 1, gt.view(-1, 1))

        prob_select_masked = - torch.masked_select(prob_select, mask.view(-1, 1).byte())
        loss = torch.mean(prob_select_masked)
        return loss


class Graph2seq(nn.Module):
    def __init__(self, vocab, hidden_size=300, direction_option='bi_sep', device='cpu'):
        super(Graph2seq, self).__init__()

        use_cuda = (device == 'cuda')
        self.vocab = vocab
        embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"}
        self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                                               vocab=vocab.in_word_vocab,
                                                               hidden_size=hidden_size, dropout=0.2, use_cuda=use_cuda,
                                                               fix_word_emb=False)
        self.gnn = None
        self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[0].word_emb_layer
        self.gnn_encoder = GAT(2, hidden_size, hidden_size, hidden_size, [2, 1], direction_option=direction_option).to(
            device)
        self.seq_decoder = StdRNNDecoder(max_decoder_step=50,
                                         decoder_input_size=2 * hidden_size if direction_option == 'bi_sep' else hidden_size,
                                         decoder_hidden_size=hidden_size,
                                         word_emb=self.word_emb, vocab=self.vocab.in_word_vocab,
                                         attention_type="sep_diff_encoder_type", fuse_strategy="concatenate",
                                         rnn_emb_input_size=hidden_size,
                                         tgt_emb_as_output_layer=True, device=self.graph_topology.device).to(device)
        self.loss_calc = Graph2seqLoss(self.vocab.in_word_vocab)

    def forward(self, graph_list, tgt=None, require_loss=True):
        batch_graph = self.graph_topology(graph_list)
        batch_graph = self.gnn_encoder(batch_graph)
        batch_graph.node_features['rnn_emb'] = batch_graph.node_features['node_feat']

        # down-task
        prob, _, _ = self.seq_decoder(from_batch(batch_graph), tgt_seq=tgt)
        if require_loss:
            loss = self.loss_calc(prob, tgt)
            return prob, loss
        else:
            return prob


class Jobs:
    def __init__(self, args):
        super(Jobs, self).__init__()
        self.device = args.device
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = JobsDataset(root_dir="graph4nlp/pytorch/test/dataset/jobs",
                              topology_builder=DependencyBasedGraphConstruction,
                              topology_subdir='DependencyGraph', share_vocab=True, val_split_ratio=0.2)

        self.train_dataloader = DataLoader(dataset.train, batch_size=24, shuffle=True, num_workers=1,
                                           collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.val, batch_size=24, shuffle=True, num_workers=1,
                                         collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=24, shuffle=True, num_workers=1,
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Graph2seq(self.vocab, device=self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=1e-3)

    def _build_evaluation(self):
        self.metrics = [ExpressionAccuracy()]

    def train(self):
        max_score = -1
        for epoch in range(200):
            self.model.train()
            print("Epoch: {}".format(epoch))
            for data in self.train_dataloader:
                graph_list, tgt = data
                tgt = tgt.to(self.device)
                _, loss = self.model(graph_list, tgt, require_loss=True)
                print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch > 4:
                score = self.evaluate(self.val_dataloader)
                if score > max_score:
                    torch.save(self.model.state_dict(), 'best_model.pt')
                max_score = max(max_score, score)
        return max_score

    def evaluate(self, dataloader):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        for data in dataloader:
            graph_list, tgt = data
            prob = self.model(graph_list, require_loss=False)
            pred = logits2seq(prob)

            pred_str = wordid2str(pred.detach().cpu(), self.vocab.in_word_vocab)
            tgt_str = wordid2str(tgt, self.vocab.in_word_vocab)
            pred_collect.extend(pred_str)
            gt_collect.extend(tgt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("accuracy: {:.3f}".format(score))
        return score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    runner = Jobs(args)
    max_score = runner.train()
    print("Train finish, best score: {:.3f}".format(max_score))
    runner.model.load_state_dict(torch.load('best_model.pt'))
    test_score = runner.evaluate(runner.test_dataloader)
    print("Test score: {:.3f}".format(test_score))
