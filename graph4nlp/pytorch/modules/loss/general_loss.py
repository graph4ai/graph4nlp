import torch.nn as nn

from .base import GeneralLossBase


class GeneralLoss(GeneralLossBase):
    r"""
        This general loss are backended on the pytorch loss function.
        The detailed decription for each loss function can be found at:
            `pytorch loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`

    Parameters
    ----------
    loss_type: str
        the loss function to select (``NLL``,``BCEWithLogits``, ``MultiLabelMargin``,``SoftMargin``
        ,``CrossEntropy`` )

        `NLL loss<https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#NLLLoss>`
        measures
        the negative log likelihood loss. It is useful to train a classification problem
        with C classes.

        `BCEWithLogits loss
        <https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss>`
        combines a
        `Sigmoid` layer and the `BCELoss` in one single class. This version is more numerically
        stable than using a plain `Sigmoid`followed by a `BCELoss` as, by combining the operations
        into one layer, we take advantage of the log-sum-exp trick for numerical stability.

        `BCE Loss<https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCELoss>`
        creates a criterion that measures
        the Binary Cross Entropy between the target and the output.

        `MultiLabelMargin loss
        <https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#MultiLabelMarginLoss>`
        creates a
        criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss)
        between input :math:`x` (a 2D mini-batch `Tensor`) and output :math:`y`
        (which is a 2D `Tensor` of target class indices).

        `SoftMargin loss
        <https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#SoftMarginLoss>`
        creates a criterion that optimizes a two-class classification logistic loss between
        input tensor :math:`x` and target tensor :math:`y` (containing 1 or -1).

        `CrossEntropy loss
        <https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss> `
        combines pytorch function `nn.LogSoftmax` and `nn.NLLLoss` in one single class. It is
        useful when training a classification problem with `C` classes.

    weight: Tensor, optional
        a manual rescaling weight given to the loss of each batch element. If given, has to be
        a Tensor of size `nbatch`.
        This parameter is not suitable for ``SoftMargin`` loss functions.
    size_average: bool, optional
        By default,the losses are averaged over each loss element in the batch. Note that for
        some losses, there are multiple elements per sample.
        If the field :attr:`size_average` is set to ``False``, the losses are instead summed
        for each minibatch. Ignored when reduce is ``False``. Default: ``True``.
    reduce: bool, optional
        By default, the losses are averaged or summed over observations for each minibatch depending
        on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction: string, optional
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of elements in the output,
          ``'sum'``: the output will be summed.
        Note: :attr:`size_average` and :attr:`reduce` are in the process of being deprecated, and
        in the meantime, specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    pos_weight:Tensor, optional
        A weight of positive examples. Must be a vector with length equal to the number of classes.
        This paramter is only suitable for ``BCEWithLogits`` loss function.
    ignore_index: int, optional
        Specifies a target value that is ignored and does not contribute to the input gradient.
        When :attr:`size_average` is ``True``, the loss is averaged over non-ignored targets.
        This paramter is only suitable for ``CrossEntropy`` loss function.
    """

    def __init__(
        self,
        loss_type,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        pos_weight=None,
    ):
        super(GeneralLoss, self).__init__()
        if loss_type == "NLL":
            self.loss_function = nn.NLLLoss(weight, size_average, ignore_index, reduce, reduction)
        if loss_type == "BCEWithLogits":
            self.loss_function = nn.BCEWithLogitsLoss(
                weight, size_average, reduce, reduction, pos_weight
            )
        if loss_type == "MultiLabelMargin":
            self.loss_function = nn.MultiLabelMarginLoss(size_average, reduce, reduction)
        if loss_type == "SoftMargin":
            self.loss_function = nn.SoftMarginLoss(size_average, reduce, reduction)
        if loss_type == "CrossEntropy":
            self.loss_function = nn.CrossEntropyLoss(
                weight, size_average, ignore_index, reduce, reduction
            )
        if loss_type == "BCE":
            self.loss_function = nn.BCELoss(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        r"""
        Compute the loss.

        Parameters
        ----------
        NLL loss:
             Input: tensor.
               :math:`(N, C)` where `C = number of classes`, or :math:`(N, C, d_1, d_2, ..., d_K)`
               with :math:`K \geq 1` in the case of `K`-dimensional loss.
             Target: tensor.
               :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
               or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional
               loss.
             Output: scalar.
               If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(N)`,
               or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional
               loss.

        BCE/BCEWithLogits loss:
             Input: Tensor.
               :math:`(N, *)` where :math:`*` means, any number of additional dimensions
             Target: Tensor.
               :math:`(N, *)`, same shape as the input
             Output: scalar.
               If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as input.

        MultiLabelMargin loss:
             Input: Tensor.
               :math:`(C)` or :math:`(N, C)` where `N` is the batch size and `C` is the number
               of classes.
             Target: Tensor.
               :math:`(C)` or :math:`(N, C)`, label targets padded by -1 ensuring same shape as
               the input.
             Output: Scalar.
                If :attr:`reduction` is ``'none'``, then :math:`(N)`.

        SoftMargin loss:
             Input: Tensor.
               :math:`(*)` where :math:`*` means, any number of additional dimensions
             Target: Tensor.
               :math:`(*)`, same shape as the input
             Output: scalar.
               If :attr:`reduction` is ``'none'``, then same shape as the input

        CrossEntropy:
             Input: Tensor.
               :math:`(N, C)` where `C = number of classes`, or :math:`(N, C, d_1, d_2, ..., d_K)`
               with :math:`K \geq 1` in the case of `K`-dimensional loss.
             Target: Tensor.
               :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
               or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional
               loss.
            Output: scalar.
               If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(N)`,
               or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional
               loss.
        """

        return self.loss_function(input, target)
