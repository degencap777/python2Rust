from enum import Enum

# MetricCategory is used to aggregate main scores at Evaluation level
class MetricCategory(Enum):
    ACCURACY=1
    BIAS=2
    TOXICITY=3

    def __repr__(self):
        return repr(self.name)