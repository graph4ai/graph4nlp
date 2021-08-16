import torch

from .base import GeneralLossBase


class CoverageLoss(GeneralLossBase):
    """
        The loss function for coverage mechanism.
    Parameters
    ----------
    cover_loss: float
        The weight for coverage loss.
    """

    def __init__(self, cover_loss):
        super(CoverageLoss, self).__init__()
        self.cover_loss = cover_loss

    def forward(self, enc_attn_weights, coverage_vectors):
        """
            The calculation function.
        Parameters
        ----------
        enc_attn_weights: list[torch.Tensor]
            The list containing all decoding steps' attention weights.
            The length should be the decoding step.
            Each element should be the tensor.
        coverage_vectors: list[torch.Tensor]
            The list containing all coverage vectors in decoding module.
        Returns
        -------
        coverage_loss: torch.Tensor
            The loss.
        """
        target_length = len(enc_attn_weights)
        loss = 0
        for i in range(target_length):
            if coverage_vectors[i] is not None:
                coverage_loss = (
                    torch.sum(torch.min(coverage_vectors[i], enc_attn_weights[i]))
                    / coverage_vectors[-1].shape[0]
                    * self.cover_loss
                )
                loss += coverage_loss
        return loss / target_length
