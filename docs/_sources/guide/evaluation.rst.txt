Chapter 7. Evaluations and Loss components
===========================
.. image:: https://img.shields.io/github/forks/graph4ai/graph4nlp?style=social
        :target: https://github.com/graph4ai/graph4nlp/fork
.. image:: https://img.shields.io/github/stars/graph4ai/graph4nlp?style=social
        :target: https://github.com/graph4ai/graph4nlp

Evaluations
--------------
This part evolves the main evaluation metrics for various tasks. These metrics derive from the save base class and have the same interface for easy use.

1) For classification problems, we implement: precision, recall, F1 and accuracy metrics

.. code:: python

    import torch
    from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
    ground_truth = torch.Tensor([0, 1, 3, 1, 1]).long()
    predict = torch.Tensor([0, 1, 3, 3, 1]).long()

    metric = Accuracy(metrics=["precision", "recall", "F1", "accuracy"])

    precision, recall, f1, accuracy = metric.calculate_scores(
            ground_truth=ground_truth, predict=predict, average="micro")

2) For generation tasks, we implement 6 metrics including: 1) BLEU (BLEU@1-4), 2), BLEUTranslation(SacreBLEU), 3)CIDEr, 4)METEOR, 5) ROUGE (ROUGE-L), 6)SummarizationRouge (implemented by pyrouge).

.. code:: python

    from graph4nlp.pytorch.modules.evaluation import BLEU

    bleu_metrics = BLEU(n_grams=[1, 2, 3, 4])

    prediction = ["I am a PHD student.", "I am interested in Graph Neural Network."]
    ground_truth = ["I am a student.", "She is interested in Math."]

    scores = bleu_metrics.calculate_scores(ground_truth=ground_truth, predict=prediction)


Loss components
----------------
We have implemented several specific loss functions for various tasks.

Sequence generation loss.
^^^^^^^^^^^^^^^^^^^
We have wrapped the cross-entropy loss and the coverage loss(optional) to calculate the final loss for various sequence generation tasks (e.g., graph2seq, seq2seq).

.. code:: python

    from graph4nlp.pytorch.modules.loss.seq_generation_loss import SeqGenerationLoss
    from graph4nlp.pytorch.models.graph2seq import Graph2Seq

    loss_function = SeqGenerationLoss(ignore_index=0, use_coverage=True)

    logits, enc_attn_weights, coverage_vectors = Graph2Seq(graph, tgt)
    graph2seq_loss = SeqGenerationLoss(logits, tgt, enc_attn_weights=enc_attn_weights, coverage_vectors=coverage_vectors)

Knowledge Graph Loss
^^^^^^^^^^^^^^^^^^^^^
In the state-of-the-art KGE models, loss functions were designed according to various
pointwise, pairwise and multi-class approaches. Refers to
`Loss Functions in Knowledge Graph Embedding Models <https://alammehwish.github.io/dl4kg-eswc/papers/paper%201.pdf>`__

**Pointwise Loss Function**

1. `MSELoss <https://pytorch.org/docs/master/generated/torch.nn.MSELoss.html>`__
creates a criterion that measures the mean squared error (squared L2 norm)
between each element in the input :math:`x` and target :math:`y`. It is the wrapper of ``nn.MSELoss`` in pytorch.


2. `SOFTMARGINLOSS <https://pytorch.org/docs/master/generated/torch.nn.SoftMarginLoss
.html>`__ Creates a criterion that optimizes a two-class classification
logistic loss between input tensor :math:`x` and target tensor :math:`y`
(containing 1 or -1). It is the wrapper of ``nn.SoftMarginLoss`` in pytorch.

The number of positive and negative samples should be about the same, otherwise it's easy to overfit.

.. math::
    \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}


**Pairwise Loss Function**

1. `SoftplusLoss <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/loss/SoftplusLoss.py>`__
refers to the paper `OpenKE: An Open Toolkit for Knowledge Embedding <https://www.aclweb.org/anthology/D18-2024.pdf>`__

.. code::

    class SoftplusLoss(nn.Module):
        def __init__(self, adv_temperature=None):
            super(SoftplusLoss, self).__init__()
            self.criterion = nn.Softplus()
            if adv_temperature != None:
                self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
                self.adv_temperature.requires_grad = False
                self.adv_flag = True
            else:
                self.adv_flag = False

        def get_weights(self, n_score):
            return torch.softmax(n_score * self.adv_temperature, dim=-1).detach()

        def forward(self, p_score, n_score):
            if self.adv_flag:
                return (self.criterion(-p_score).mean() + (self.get_weights(n_score) * self.criterion(n_score)).sum(
                    dim=-1).mean()) / 2
            else:
                return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2

2. `SigmoidLoss <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/loss/SigmoidLoss.py>`__
refers to the paper `OpenKE: An Open Toolkit for Knowledge Embedding <https://www.aclweb.org/anthology/D18-2024.pdf>`__

.. code::

    class SigmoidLoss(nn.Module):
        def __init__(self, adv_temperature = None):
            super(SigmoidLoss, self).__init__()
            self.criterion = nn.LogSigmoid()
            if adv_temperature != None:
                self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
                self.adv_temperature.requires_grad = False
                self.adv_flag = True
            else:
                self.adv_flag = False

        def get_weights(self, n_score):
            return torch.softmax(n_score * self.adv_temperature, dim = -1).detach()

        def forward(self, p_score, n_score):
            if self.adv_flag:
                return -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2
            else:
                return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2

The implementations of ``SoftplusLoss`` and ``SigmoidLoss`` refer to `OpenKE <https://github.com/thunlp/OpenKE>`__.

**Multi-Class Loss Function**

1. `Binary Cross Entropy Loss <https://pytorch.org/docs/master/generated/torch.nn.BCELoss.html>`__
Creates a criterion that measures the Binary Cross Entropy between the target and the output. Note that the targets
:math:`y` should be numbers between 0 and 1. It is the wrapper of ``nn.BCELoss`` in pytorch.

Next it is a simple how to use code:

.. code:: python

    import torch
    from graph4nlp.pytorch.modules.loss.kg_loss import KGLoss

    loss_function = KGLoss(loss_type="BCELoss")
    m = nn.Sigmoid()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss_function(m(input), target)


General Loss
^^^^^^^^^^^^^^^^^^^^^
It includes the most used loss functions containing:

1) ``NLL`` loss. It is the wrapper of ``nn.NLLLoss`` in pytorch.

2) ``BCE`` loss. It is the wrapper of ``nn.BCELoss`` in pytorch.

3) ``BCEWithLogits`` loss. It is the wrapper of ``nn.BCEWithLogitsLoss`` in pytorch.

4) ``MultiLabelMargin`` loss. It is the wrapper of ``nn.MultiLabelMarginLoss`` in pytorch.

5) ``SoftMargin`` loss. It is the wrapper of ``nn.SoftMargin`` in pytorch.

6) ``CrossEntropy`` loss. It is the wrapper of ``nn.CrossEntropy`` in pytorch.

Next it is a simple how to use code:

.. code:: python

    import torch
    from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss

    loss_function = GeneralLoss(loss_type="CrossEntropy")
    input = torch.randn(3, 5)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss_function(input, target)
