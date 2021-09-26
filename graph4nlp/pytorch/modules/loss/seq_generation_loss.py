import torch

from .base import GeneralLossBase
from .coverage_loss import CoverageLoss
from .general_loss import GeneralLoss


class SeqGenerationLoss(GeneralLossBase):
    """
        The general loss for ``Graph2Seq`` model.
    Parameters
    ----------
    ignore_index: ignore_index
        The token index which will be ignored during calculation. Usually it is the padding index.
    use_coverage: bool, default=False
        Whether use coverage mechanism. If set ``True``, the we will add the coverage loss.
    coverage_weight: float, default=0.3
        The weight of coverage loss.
    """

    def __init__(self, ignore_index, use_coverage=False, coverage_weight=0.3):
        super(SeqGenerationLoss, self).__init__()
        self.use_coverage = use_coverage
        self.loss_ce = GeneralLoss(loss_type="NLL", reduction="mean", ignore_index=ignore_index)
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
            loss_cover = self.loss_coverage(enc_attn_weights, coverage_vectors)
            loss_ce += loss_cover
        return loss_ce
