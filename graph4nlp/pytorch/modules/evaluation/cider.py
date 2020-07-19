from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.evaluation.cider_tool.cider import Cider


class CIDEr(EvaluationMetricBase):
    """
        The CIDEr evaluation metric class.
    Parameters
    ----------
    df: string
        Parameter indicating document frequency.
    """

    def __init__(self, df):
        super(CIDEr, self).__init__()
        self.scorer = Cider(df=df)

    def calculate_scores(self, ground_truth, predict):
        """
            The CIDEr calculation function. It will compute the CIDEr scores.
        Parameters
        ----------
        ground_truth: list[string]
            The ground truth (correct) target values. It is a list of strings.
        predict: list[string]
            The predicted target values. It is a list of strings.
        Returns
        -------
        score: float
            The CIDEr value.
        scores: list[float]
            The specific results for CIDEr metric.
        """
        ref_list = [list(map(str.strip, refs)) for refs in zip(ground_truth)]

        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(predict)}
        assert len(refs) == len(hyps)
        score, scores = self.scorer.compute_score(refs, hyps)
        return score, scores
