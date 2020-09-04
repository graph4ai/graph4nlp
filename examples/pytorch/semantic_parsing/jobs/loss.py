import torch
import torch.nn as nn

from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab


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

        prob_select_masked = - torch.masked_select(prob_select, mask.view(-1, 1).bool())
        loss = torch.mean(prob_select_masked)
        return loss


class CoverageLoss(nn.Module):
    def __init__(self, cover_loss):
        super(CoverageLoss, self).__init__()
        self.cover_loss = cover_loss

    def forward(self, batch_size, enc_attn_weights, coverage_vectors):
        target_length = len(enc_attn_weights)
        loss = 0
        for i in range(target_length):
            if coverage_vectors[i] is not None:
                coverage_loss = torch.sum(
                    torch.min(coverage_vectors[i], enc_attn_weights[i])) / batch_size * self.cover_loss
                loss += coverage_loss
        return loss / target_length
