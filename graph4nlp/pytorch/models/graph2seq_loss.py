import torch.nn as nn

from ..modules.loss.seq_generation_loss import SeqGenerationLoss


class Graph2SeqLoss(nn.Module):
    def __init__(self, ignore_index, use_coverage, coverage_weight):
        super(Graph2SeqLoss, self).__init__()
        self.loss_func = SeqGenerationLoss(
            ignore_index=ignore_index, use_coverage=use_coverage, coverage_weight=coverage_weight
        )

    def forward(self, logits, label, enc_attn_weights=None, coverage_vectors=None):
        loss = self.loss_func(
            logits=logits,
            label=label,
            enc_attn_weights=enc_attn_weights,
            coverage_vectors=coverage_vectors,
        )
        return loss
