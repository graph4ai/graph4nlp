import torch.nn as nn
import torch

from graph4nlp.pytorch.modules.loss.coverage_loss import CoverageLoss
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.loss.base import GeneralLossBase
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab

class GraphGMP2SeqLoss(nn.Module):
    def __init__(self, ignore_index, use_coverage, coverage_weight):
        super(GraphGMP2SeqLoss, self).__init__()
        # self.loss_func = SeqGenerationLoss(ignore_index=ignore_index, use_coverage=use_coverage,
        #                                    coverage_weight=coverage_weight)

        self.use_coverage = use_coverage
        self.loss_ce = GeneralLoss(loss_type="NLL", size_average=True, reduce=True,
                                   ignore_index=ignore_index)
        self.loss_coverage = CoverageLoss(cover_loss=coverage_weight)

    def forward(self, logits, label, enc_attn_weights=None, coverage_vectors=None):
        """
            The calculation method.
        Parameters
        ----------
        logits: torch.Tensor
            The probability with the shape of ``[batch_size, max_decoder_step, vocab_size]``. \
            Note that it is calculated by ``softmax``.

        label: torch.Tensor
            The ground-truth with the shape of ``[batch_size, max_decoder_step]``.
        enc_attn_weights: list[torch.Tensor], default=None
            The list containing all decoding steps' attention weights.
            The length should be the decoding step.
            Each element should be the tensor.
        coverage_vectors: list[torch.Tensor], default=None
            The list containing all coverage vectors in decoding module.

        Returns
        -------
        graph2seq_loss: torch.Tensor
        """
        loss_ce = self.loss_ce(torch.log(logits + 1e-31).transpose(1, 2), label)
        if self.use_coverage:
            loss_cover_0 = self.loss_coverage(enc_attn_weights[0], coverage_vectors[0])
            loss_cover_1 = self.loss_coverage(enc_attn_weights[1], coverage_vectors[1])
            loss_ce += (loss_cover_0 + loss_cover_1)
        return loss_ce