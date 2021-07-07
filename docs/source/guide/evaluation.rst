Chapter 6. Evaluations
===========================
This part evolves the main evaluation metrics for various tasks. Specifically, we implement 7 metrics including: 1) accuracy (precision, recall, F1), 2) BLEU (BLEU@1-4), 3), BLEUTranslation(SacreBLEU), 4)CIDEr, 5)METEOR, 6) ROUGE (ROUGE-L), 7)SummarizationRouge (implemented by pyrouge). These metrics derive from the save base class and have the same interface for easy use.


.. code:: python

    from graph4nlp.pytorch.modules.evaluation import BLEU

    bleu_metrics = BLEU(n_grams=[1, 2, 3, 4])

    prediction = ["I am a PHD student.", "I am interested in Graph Neural Network."]
    ground_truth = ["I am a student.", "She is interested in Math."]

    scores = bleu_metrics.calculate_scores(ground_truth=ground_truth, predict=prediction)