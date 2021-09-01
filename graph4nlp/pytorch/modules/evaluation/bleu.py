from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.evaluation.bleu_tool.bleu import Bleu


class BLEU(EvaluationMetricBase):
    """
        The BLEU evaluation metric class.
    Parameters
    ----------
    n_grams: list[int]
        The BLEU's n_gram parameter. The results will be returned according \
            to the ``n_grams`` one-by-one.
    verbose: int, default = 0
        The log indicator. If set to 0, it will output no logs.
    """

    def __init__(self, n_grams, verbose=0):
        super(BLEU, self).__init__()
        max_gram = self._check_available(n_grams)
        self.scorer = Bleu(max_gram, verbose=verbose)
        self.metrics = n_grams

    def calculate_scores(self, ground_truth, predict):
        """
            The BLEU calculation function. It will compute the BLEU scores.
        Parameters
        ----------
        ground_truth: list[string]
            The ground truth (correct) target values. It is a list of strings.
        predict: list[string]
            The predicted target values. It is a list of strings.
        Returns
        -------
        score: list[float]
            The list contains BLEU_n results according to ``n_grams``.
        scores: list[list[float]]
            The specific results for each needed BLEU_n metric.
        """
        ref_list = [list(map(str.strip, refs)) for refs in zip(ground_truth)]

        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(predict)}
        assert len(refs) == len(hyps)
        score, scores = self.scorer.compute_score(refs, hyps)
        score_ret = [score[i - 1] for i in self.metrics]
        scores_ret = [scores[i - 1] for i in self.metrics]
        return score_ret, scores_ret

    @staticmethod
    def _check_available(n_grams):
        """
            The function to check the parameters.
            If all tests are passed, it will find the max value in ``n_grams``.
        Parameters
        ----------
        n_grams: list[int]

        Returns
        -------
        max_n_grams_value: int
        """
        n_grams_ok = True
        if isinstance(n_grams, list):
            for i in n_grams:
                if not isinstance(i, int):
                    n_grams_ok = False
        else:
            n_grams_ok = False
        if not n_grams_ok:
            raise TypeError("argument n_grams must be in list of int.")
        return max(n_grams)
