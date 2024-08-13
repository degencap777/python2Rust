from core.interaction import Interaction
from utils.numeric_sequence_stats import NumericSequenceStats

class TaskExecutionStats():
    def __init__(self, interactions: list[Interaction], score: float, task_duration_seconds: float):
        self.scores = NumericSequenceStats()
        self.scores.add_value(score)
        self.durations = NumericSequenceStats()
        self.durations.add_value(task_duration_seconds)
        self.num_inferences = NumericSequenceStats()
        self.num_inferences.add_value(len(interactions))
        self.inference_latencies = NumericSequenceStats()
        self.cached_inference_latencies = NumericSequenceStats()
        self.cache_read_latencies = NumericSequenceStats()
        self.cache_write_latencies = NumericSequenceStats()
        self.prompt_lengths = NumericSequenceStats()
        self.output_lengths = NumericSequenceStats()
        for interaction in interactions:
            if interaction.flags.cache_hit:
                self.cached_inference_latencies.add_value(interaction.timers.cache_read_latency.seconds)
            else:
                self.inference_latencies.add_value(interaction.timers.inference_latency.seconds)
            self.cache_read_latencies.add_value(interaction.timers.cache_read_latency.seconds)
            self.cache_write_latencies.add_value(interaction.timers.cache_write_latency.seconds)
            self.prompt_lengths.add_value(len(interaction.prompt.to_text()))
            self.output_lengths.add_value(len(interaction.output.to_text()))

    @staticmethod
    def aggregate(runs: list['TaskExecutionStats']) -> 'TaskExecutionStats':
        agg = TaskExecutionStats([], 0.0, 0.0)
        # Reset the 3 fields that get auto-assigned elements
        agg.scores.values = []
        agg.durations.values = []
        agg.num_inferences.values = []
        for run in runs:
            agg.scores.add_values(run.scores.values)
            agg.durations.add_values(run.durations.values)
            agg.num_inferences.add_values(run.num_inferences.values)
            agg.inference_latencies.add_values(run.inference_latencies.values)
            agg.cached_inference_latencies.add_values(run.cached_inference_latencies.values)
            agg.cache_read_latencies.add_values(run.cache_read_latencies.values)
            agg.cache_write_latencies.add_values(run.cache_write_latencies.values)
            agg.prompt_lengths.add_values(run.prompt_lengths.values)
            agg.output_lengths.add_values(run.output_lengths.values)
        return agg
   