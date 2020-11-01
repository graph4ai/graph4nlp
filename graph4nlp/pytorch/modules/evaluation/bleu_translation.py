from .base import EvaluationMetricBase


class BLEUTranslation(EvaluationMetricBase):
    def __init__(self):
        super(BLEUTranslation, self).__init__()

    def calculate_scores(self, **kwargs):
        pass