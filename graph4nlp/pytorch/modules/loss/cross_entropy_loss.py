import torch
import torch.nn as nn

from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab

from .base import GeneralLossBase


class CrossEntropyLoss(GeneralLossBase):
    """
       The cross-entropy with ``mask`` support.
       Technically, it is a wrapper of ``nn.NLLLoss``.

    Parameters
    ----------
    vocab: Vocab
       The vocab for loss calculation.
    """

    def __init__(self, vocab: Vocab):
        super(CrossEntropyLoss, self).__init__()
        self.loss_func = nn.NLLLoss()
        self.VERY_SMALL_NUMBER = 1e-31
        self.vocab = vocab

    def forward(self, logits, label):
        """
            The loss calculation.
        Parameters
        ----------
        logits: torch.Tensor
            The probability with the shape of ``[batch_size, max_decoder_step, vocab_size]``. \
            Note that it is calculated by ``softmax``.

        label: torch.Tensor
            The ground-truth with the shape of ``[batch_size, max_decoder_step]``.

        Returns
        -------
        ce_loss: torch.Tensor
            The cross-entropy loss.
        """
        assert logits.shape[0:1] == label.shape[0:1]
        assert len(logits.shape) == 3
        log_prob = torch.log(logits + self.VERY_SMALL_NUMBER)
        batch_size = label.shape[0]
        step = label.shape[1]

        mask = 1 - label.data.eq(self.vocab.PAD).float()

        prob_select = torch.gather(log_prob.view(batch_size * step, -1), 1, label.view(-1, 1))

        prob_select_masked = -torch.masked_select(prob_select, mask.view(-1, 1).bool())
        loss = torch.mean(prob_select_masked)
        return loss
