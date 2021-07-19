Chapter 7. Evaluations and Loss components
===========================

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
    output = loss(input, target)
