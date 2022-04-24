from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase


class ExpressionAccuracy(EvaluationMetricBase):
    def __init__(self):
        super(ExpressionAccuracy, self).__init__()

    def calculate_scores(self, ground_truth, predict):
        correct = 0
        assert len(ground_truth) == len(predict)
        for gt, pred in zip(ground_truth, predict):
            if gt == pred:
                correct += 1.0
        return correct / len(ground_truth)
