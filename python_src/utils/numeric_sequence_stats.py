from time import time
from statistics import mean, median
from utils.numerals import NumberUtils

class NumericSequenceStats():
    def __init__(self) -> None:
        self.values = []
        self.creation_time = time()

    def add_value(self, value):
        if value is not None:
            self.values.append(float(value))

    def add_values(self, values: list):
        for value in values:
            if value is not None:
                self.values.append(float(value))

    def count(self) -> int:
        return len(self.values)

    def mean(self) -> float:
        if not len(self.values):
            return 0.0
        return NumberUtils.round_float(mean(self.values))

    def median(self) -> float:
        if not len(self.values):
            return 0.0
        return NumberUtils.round_float(median(self.values))

    def max(self) -> float:
        return max(self.values) if len(self.values) else 0.0

    def min(self) -> float:
        return min(self.values) if len(self.values) else 0.0

    def sum(self) -> float:
        return sum(self.values) if len(self.values) else 0.0

    def age_seconds(self) -> float:
        return time() - self.creation_time