class EvaluationMetricBase:
    """
    Base class for evaluation metric
    """

    def __init__(self):
        pass

    def calculate_scores(self, **kwargs):
        raise NotImplementedError()
