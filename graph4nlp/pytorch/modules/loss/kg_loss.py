import torch
import torch.nn as nn

from .base import KGLossBase


class SigmoidLoss(nn.Module):
    r"""
    Pairwise loss function. :math:`x_p` is the predition score of positive examples,
    and :math:`x_n` is the predition score of negative examples.

    .. math::
        \text{loss}(x_p, x_n) = -{\sum_i log \frac{1} {1+\exp(-x_p[i])} +
        \sum_j \frac{1} log(1+\exp(x_n[j])))
    """

    def __init__(self, adv_temperature=None):
        super(SigmoidLoss, self).__init__()
        self.criterion = nn.LogSigmoid()
        if adv_temperature is not None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return torch.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (
                -(
                    self.criterion(p_score).mean()
                    + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim=-1).mean()
                )
                / 2
            )
        else:
            return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2


class SoftplusLoss(nn.Module):
    r"""
    Pairwise loss function. :math:`x_p` is the predition score of positive examples,
    and :math:`x_n` is the predition score of negative examples.

    .. math::
        \text{loss}(x_p, x_n) = \sum_i log(1+\exp(-x_p[i])) + \sum_j log(1+\exp(x_n[j]))
    """

    def __init__(self, adv_temperature=None):
        super(SoftplusLoss, self).__init__()
        self.criterion = nn.Softplus()
        if adv_temperature is not None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return torch.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (
                self.criterion(-p_score).mean()
                + (self.get_weights(n_score) * self.criterion(n_score)).sum(dim=-1).mean()
            ) / 2
        else:
            return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2


class KGLoss(KGLossBase):
    r"""
    In the state-of-the-art KGE models, loss functions were designed according to
    various pointwise, pairwise and multi-class approaches. Refers to
    `Loss Functions in Knowledge Graph Embedding Models
    <https://alammehwish.github.io/dl4kg-eswc/papers/paper%201.pdf>`__

    **Pointwise Loss Function**

    `MSELoss <https://pytorch.org/docs/master/generated/torch.nn.MSELoss.html>`__
    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input :math:`x` and target :math:`y`.


    `SOFTMARGINLOSS <https://pytorch.org/docs/master/generated/torch.nn.SoftMarginLoss
    .html>`__ Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor :math:`x` and target tensor :math:`y`
    (containing 1 or -1).
    Tips:   The number of positive and negative samples should be about the same,
            otherwise it's easy to overfit

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}


    **Pairwise Loss Function**

    `SoftplusLoss <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/loss/
    SoftplusLoss.py>`__
    refers to the paper `OpenKE: An Open Toolkit for Knowledge Embedding
    <https://www.aclweb.org/anthology/D18-2024.pdf>`__

    `SigmoidLoss <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/loss/
    SigmoidLoss.py>`__
    refers to the paper `OpenKE: An Open Toolkit for Knowledge Embedding
    <https://www.aclweb.org/anthology/D18-2024.pdf>`__

    **Multi-Class Loss Function**

    `Binary Cross Entropy Loss <https://pytorch.org/docs/master/generated/torch.nn.BCELoss.html>`__
    Creates a criterion that measures the Binary Cross Entropy between the target
    and the output. Note that the targets
    :math:`y` should be numbers between 0 and 1.

    """

    def __init__(
        self,
        loss_type,
        size_average=None,
        reduce=None,
        reduction="mean",
        adv_temperature=None,
        weight=None,
    ):
        super(KGLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "MSELoss":
            self.loss_function = nn.MSELoss(size_average, reduce, reduction)
        elif loss_type == "SoftMarginLoss":
            self.loss_function = nn.SoftMarginLoss(size_average, reduce, reduction)
        elif loss_type == "SoftplusLoss":
            self.loss_function = SoftplusLoss(adv_temperature)
        elif loss_type == "SigmoidLoss":
            self.loss_function = SigmoidLoss(adv_temperature)
        elif loss_type == "BCELoss":
            self.loss_function = nn.BCELoss(weight, size_average, reduce, reduction)
        else:
            raise NotImplementedError()
        return

    def forward(self, input=None, target=None, p_score=None, n_score=None):
        """

        Parameters
        ----------
        MSELoss
            input: Tensor.
                :math:`(N,*)` where :math:`*` means any number of additional dimensions
            target: Tensor.
                :math:`(N,*)`, same shape as the input
            output:
                If reduction is `'none'`, then same shape as the input

        SoftMarginLoss
            input: Tensor.
                :math:`(*)` where * means, any number of additional dimensions
            target: Tensor.
                same shape as the input
            output: scalar.
                If reduction is `'none'`, then same shape as the input

        SoftplusLoss
            p_score: Tensor.
                :math:`(*)` where * means, any number of additional dimensions
            n_score: Tensor.
                :math:`(*)` where * means, any number of additional dimensions.
                The dimension could be different from the `p_score` dimension.
            output: scalar.

        SigmoidLoss
            p_score: Tensor.
                :math:`(*)` where * means, any number of additional dimensions
            n_score: Tensor.
                :math:`(*)` where * means, any number of additional dimensions.
                The dimension could be different from the `p_score` dimension.
            output: scalar.

        BCELoss:
            Input: Tensor.
                :math:`(N, *)` where :math:`*` means, any number of additional dimensions
            Target: Tensor.
                :math:`(N, *)`, same shape as the input
            Output: scalar.
                If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as input.

        Returns
        -------

        """
        if self.loss_type in ["SoftplusLoss", "SigmoidLoss"]:
            return self.loss_function(p_score, n_score)
        else:
            return self.loss_function(input, target)
