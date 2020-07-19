from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase
from graph4nlp.pytorch.modules.evaluation.meteor_tool.meteor import Meteor


class METEOR(EvaluationMetricBase):
    """
        The METEOR evaluation metric class.
    """

    def __init__(self):
        """
            The initial method for METEOR class
        Parameters
        ----------
        df: string
            Parameter indicating document frequency.
        """
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


if __name__ == "__main__":
    import json

    scorer = METEOR()
    pred_file_path = "/home/shiina/shiina/question/iq/pred.json"
    gt_file_path = "/home/shiina/shiina/question/iq/gt.json"
    with open(gt_file_path, "r") as f:
        gt = json.load(f)
        print(gt[0])
        gts = []
        for i in gt:
            for j in i:
                gts.append(str(j))
    with open(pred_file_path, "r") as f:
        pred = json.load(f)
        print(pred[1])
        preds = []
        for i in pred:
            for j in i:
                preds.append(str(j))
    print(len(gts), len(preds))
    score, scores = scorer.calculate_scores(gts, preds)
    print(score)
    print(len(scores))
