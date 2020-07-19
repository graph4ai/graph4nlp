from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.evaluation.meteor_tool.meteor import Meteor


class METEOR(EvaluationMetricBase):
    """
        The METEOR evaluation metric class.
    Parameters
    ----------
    """

    def __init__(self):
        super(METEOR, self).__init__()
        self.scorer = Meteor()

    def calculate_scores(self, ground_truth, predict):
        """
            The METEOR calculation function. It will compute the METEOR scores.
        Parameters
        ----------
        ground_truth: list[string]
            The ground truth (correct) target values. It is a list of strings.
        predict: list[string]
            The predicted target values. It is a list of strings.
        Returns
        -------
        score: float
            The METEOR value.
        scores: list[float]
            The specific results for METEOR metric.
        """
        ref_list = [list(map(str.strip, refs)) for refs in zip(ground_truth)]

        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(predict)}
        assert len(refs) == len(hyps)
        score, scores = self.scorer.compute_score(refs, hyps)
        return score, scores
