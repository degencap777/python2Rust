from core.metric_category import MetricCategory

# Evaluator converts this Score object to a denormalized TaskScore stored in Spanner.
class Score():
    def __init__(self, metric_category: MetricCategory, metric_name: str, score: float, rater_name: str = 'Proton', rater_version: str = 'V1' ):
        self.metric_name = metric_name
        self.metric_category = metric_category
        self.score = score
        self.rater_name = rater_name
        self.rater_version = rater_version

    def __repr__(self):
        return f"{self.metric_name}: {self.score * 100:.1f}%"
