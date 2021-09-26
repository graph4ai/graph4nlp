import sacrebleu

from .base import EvaluationMetricBase


class BLEUTranslation(EvaluationMetricBase):
    def __init__(self):
        super(BLEUTranslation, self).__init__()

    def calculate_scores(self, ground_truth, predict):
        """
            The standard BLEU calculation function for translation. It will compute the BLEU \
                scores using sacrebleu tools.
        Parameters
        ----------
        ground_truth: list[string]
            The ground truth (correct) target values. It is a list of strings.
        predict: list[string]
            The predicted target values. It is a list of strings.
        Returns
        -------
        score: float
            The final bleu score
        """
        bleu = sacrebleu.corpus_bleu(predict, [ground_truth], lowercase=True)
        return bleu.score
