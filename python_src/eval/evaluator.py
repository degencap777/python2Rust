import random
from enum import Enum
from datetime import datetime

from core.task import Task
from core.batch_task import BatchTask
from core.data_bundle import DataBundle
from core.conversation import Conversation
from core.interaction import Interaction
from core.batch import Batch
from core.score import Score

from eval.eval_tracker import EvalTracker
from eval.dataset import Dataset
from eval.eval_run import EvalRun
from eval.task_execution_stats import TaskExecutionStats
from eval.task_execution import TaskExecution
from eval.primitive_execution import PrimitiveExecution
from eval.sampling_method import SamplingMethod

from engine.engine import Engine
from engine.batch_execution import BatchExecution

from utils.logger import Trace, log
from utils.exceptions import InvalidArgsError
from utils.timer import Timer
from utils.strings import StringUtils

class TrackingMode(Enum):
    NONE=0
    FULL=1
    RANDOM_SAMPLE_SINGLE=2
    RANDOM_SAMPLE_DOZEN=3

class Evaluator():
    """Orchestrates evaluation runs."""

    def __init__(
        self,
        engine: Engine,
        spanner_project_id: str | None,
        spanner_instance_id: str | None,
        database_id: str | None,
        namespace: str = 'DEFAULT',
        trace: Trace = Trace.OFF
    ):
        if not spanner_project_id:
            raise InvalidArgsError('spanner_project_id is required.')
        if not spanner_instance_id:
            raise InvalidArgsError('spanner_instance_id is required.')
        if not database_id:
            raise InvalidArgsError('database_id is required.')
        self.engine = engine
        self.tracker = EvalTracker(spanner_project_id, spanner_instance_id, database_id, namespace)
        self.trace = trace

    def evaluate(
            self,
            tasks: Task | BatchTask | list[Task] | list[BatchTask],
            datasets: Dataset | list[Dataset],
            eval_name: str = '',
            eval_version: str = '',
            model_name: str = '',
            model_version: str = '',
            dataset_location: str = '',
            dataset_version: str = '',
            sampling_method: SamplingMethod = SamplingMethod.NONE,
            sample_size: int = 0,
            inference_tracking: TrackingMode = TrackingMode.NONE,  # Full tracking increases latency and Spanner costs.
            use_short_docs: bool = False,  # Passed to each task.run
            trace: Trace = Trace.OFF
        ) -> EvalRun:
        """The lists of Tasks and Dataset must match by index. All other parameters are optional."""
        if not tasks:
            raise InvalidArgsError('Tasks list is empty.')
        if not datasets:
            raise InvalidArgsError('Datasets list is empty.')
        self.trace = trace
        self.tracker.trace = trace
        if isinstance(tasks, Task):
            tasks = [tasks]
        elif isinstance(tasks, BatchTask):
            tasks = [tasks]
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        if len(tasks) != len(datasets):
            raise InvalidArgsError('Tasks and Datasets lists must match by index.')
        if not eval_name:
            eval_name = f'{len(tasks)}_tasks' if len(tasks) > 1 else tasks[0].task_id
        if not eval_version:
            eval_version = datetime.strftime(datetime.now(), '%b%d_%H:%M:%S')
        if not model_name:
            model_name, version = self.engine.default_model().parse_model_name_version()
            if not model_version:
                model_version = version
        first_dataset_length = len(datasets[0])
        comment = f'up to {sample_size} inputs' if sample_size else f'{first_dataset_length} inputs'
        log(f'Evaluating {len(tasks)} tasks on {comment} using {self.engine.default_model().model_name}.')
        self.inference_tracking = inference_tracking
        self.trace = trace
        self.tracker.start_eval_run(eval_name, eval_version, model_name, model_version, dataset_location, dataset_version)
        duration = Timer()
        task_run_stats = []
        for task, dataset in zip(tasks, datasets):
            stats = self.track_task_run(task, dataset, dataset_location, dataset_version, sampling_method, sample_size, use_short_docs)
            task_run_stats.extend(stats)
        agg = TaskExecutionStats.aggregate(task_run_stats)
        scaler = self.engine.scaler
        inference_stats = f', {scaler.num_inferences} inferences, {scaler.num_input_tokens + scaler.num_output_tokens} tokens' if self.trace > 0 else ''
        evaluation_duration = duration.seconds
        self.tracker.finalize_eval_run(scaler, evaluation_duration, len(tasks), agg.scores.mean(), agg.scores.median(), inference_stats)
        self.display_eval_summary(agg, eval_name, eval_version, len(tasks), sample_size, first_dataset_length, evaluation_duration, inference_stats, agg.prompt_lengths.mean())
        return self.tracker.eval_run

    def track_task_run(self, task: Task | BatchTask, dataset: Dataset, dataset_location, dataset_version, sampling_method: SamplingMethod = SamplingMethod.NONE, sample_size: int = 0, use_short_docs: bool = False) -> list[TaskExecutionStats]:
        self.engine.flags.shortdocs_enabled = use_short_docs
        dataset = dataset.get_sample(sampling_method, sample_size)
        self.tracker.start_task_run(task.task_id, task.task_version, dataset_location, dataset_version or 'V1.0')
        duration = Timer()
        if isinstance(task, BatchTask):
            task_run_stats = self.run_batch_task_on_dataset(task, dataset)
        else:
            task_run_stats = self.run_single_input_task_on_dataset(task, dataset)
        agg = TaskExecutionStats.aggregate(task_run_stats)
        self.tracker.finalize_task_run(
            duration = duration.seconds,
            num_executions = len(dataset),
            mean_execution_latency = agg.durations.mean(),
            median_execution_latency = agg.durations.median(),
            min_execution_latency = agg.durations.min(),
            max_execution_latency = agg.durations.max(),
            num_inferences = agg.num_inferences.count(),
            mean_inference_latency = agg.inference_latencies.mean(),
            median_inference_latency = agg.inference_latencies.median(),
            num_cached_inferences = agg.cached_inference_latencies.count(),
            num_cache_reads = agg.cache_read_latencies.count(),
            mean_cache_read_latency = agg.cache_read_latencies.mean(),
            median_cache_read_latency = agg.cache_read_latencies.median(),
            num_cache_writes = agg.cache_write_latencies.count(),
            mean_cache_write_latency = agg.cache_write_latencies.mean(),
            median_cache_write_latency = agg.cache_write_latencies.median(),
            mean_prompt_length = agg.prompt_lengths.mean(),
            mean_prediction_length = agg.output_lengths.mean(),
            mean_score = agg.scores.mean(),
            median_score = agg.scores.median()
        )
        return task_run_stats

    def run_single_input_task_on_dataset(self, task: Task, dataset: Dataset) -> list[TaskExecutionStats]:
        if not len(dataset):
            raise InvalidArgsError('Dataset is empty.')
        task_run_stats: list[TaskExecutionStats] = []
        log(f'Evaluating task {task.task_id} on {len(dataset)} ground truth records.')
        for bundle, ground_truth, metadata, cached_result in dataset:
            if isinstance(bundle, list) and len(bundle) == 1 and isinstance(bundle[0], DataBundle):
                bundle = bundle[0]
            if not isinstance(bundle, DataBundle):
                raise InvalidArgsError(f'Evaluation dataset for single input tasks must yield DataBundle (received {type(bundle)})')
            interaction = Interaction(Conversation.from_data_bundle(bundle), data_source=bundle.data_source)
            stats = self.execute_task(task, interaction, ground_truth, bundle.to_text(), metadata, cached_result)
            task_run_stats.append(stats)
        return task_run_stats

    def run_batch_task_on_dataset(self, task: BatchTask, dataset: Dataset) -> list[TaskExecutionStats]:
        if not len(dataset):
            raise InvalidArgsError('Dataset is empty.')
        task_run_stats: list[TaskExecutionStats] = []
        log(f'Evaluating batch task {task.task_id} on {len(dataset)} ground truth records.')
        for bundles, ground_truth, metadata, cached_result in dataset:
            if not isinstance(bundles, list):
                raise InvalidArgsError('Evaluation dataset for batch tasks must yield a list[DataBundle]')
            batch = Batch.from_data_bundles(bundles)
            context_for_tracing = batch.combine_interaction_contexts(defrag_text=True).to_text()
            stats = self.execute_task(task, batch, ground_truth, context_for_tracing, metadata, cached_result)
            task_run_stats.append(stats)
        return task_run_stats

    def execute_task(self, task: BatchTask | Task, context: Batch | Interaction, ground_truth: DataBundle, input_as_text: str, metadata: dict, cached_result: DataBundle | None) -> TaskExecutionStats:
        timer = Timer()
        data_source = f'({context.data_source.name})' if context.data_source and context.data_source.name else ''
        log(f'{task.task_id}{data_source}')
        if self.inference_tracking != TrackingMode.NONE:
            self.engine.enable_token_stats()  # Disable by default because of latency overhead in Bison
        self.engine.start_tracking()
        self.engine.scaler.start_inference_tracking()
        if cached_result:
            log(f'evaluator is skipping task execution for {task.task_id} because this dataset provided a cached result.')
            result = cached_result
        else:
            result = task.run(self.engine, context) # type: ignore
        timer.stop()
        interactions = self.engine.scaler.stop_inference_tracking()
        primitive_executions = self.engine.stop_tracking()
        scoring_result = task.rate(result, ground_truth)  # The actual scoring happens here
        main_score = scoring_result if isinstance(scoring_result, Score) else scoring_result[0]
        all_scores = [main_score] if isinstance(scoring_result, Score) else scoring_result
        if len(str(result)) > 160 and self.trace != Trace.ON:
            log(f'SCORE={main_score.score:.1f}\nRESULT={StringUtils.truncate(str(result), 160)}\nLABEL= {StringUtils.truncate(str(ground_truth), 160)}')
        elif len(str(result)) > 60:
            log(f'SCORE={main_score.score:.1f}\nresult={result}\nlabel= {ground_truth}')
        else:
            log(f'SCORE={main_score.score:.1f} result={result} label={ground_truth}')
        ts = TaskExecutionStats(interactions, main_score.score, timer.seconds)
        task_execution = self.track_task_execution(ground_truth, input_as_text, metadata, timer.seconds, result, interactions, primitive_executions, main_score, ts)
        for score in all_scores:
            self.tracker.create_task_score(
                task_execution_id = task_execution.id,
                rater_name = score.rater_name,
                rater_version = score.rater_version,
                metric_category = str(score.metric_category),
                metric_name = score.metric_name,
                is_main_metric = True,
                score = score.score
            )
        return ts

    def track_task_execution(self, ground_truth: DataBundle, input_as_text: str, metadata: dict, duration_seconds: float, task_result: DataBundle, interactions: list[Interaction], primitive_executions: list[BatchExecution], score: Score, stats: TaskExecutionStats) -> TaskExecution:
        task_execution = self.tracker.create_task_execution(
            duration = duration_seconds,
            num_inferences = len(interactions),
            mean_inference_latency = stats.inference_latencies.mean(),
            median_inference_latency = stats.inference_latencies.median(),
            num_cached_inferences = stats.cached_inference_latencies.count(),
            num_cache_reads = stats.cache_read_latencies.count(),
            mean_cache_read_latency = stats.cache_read_latencies.mean(),
            median_cache_read_latency = stats.cache_read_latencies.median(),
            num_cache_writes = stats.cache_write_latencies.count(),
            mean_cache_write_latency = stats.cache_write_latencies.mean(),
            median_cache_write_latency = stats.cache_write_latencies.median(),
            mean_prompt_length = stats.prompt_lengths.mean(),
            mean_prediction_length = stats.output_lengths.mean(),
            input_text = input_as_text,
            input_location = metadata['input_location'] if 'input_location' in metadata else '',
            output_text = task_result.to_text(),
            output_location = metadata['output_location'] if 'output_location' in metadata else '',
            output_context = interactions[-1].prompt.to_text() if interactions else '', # Last executed prompt for investigations and SFT
            last_inference_output = interactions[-1].output.to_text() if interactions else '',
            ground_truth = ground_truth.to_text(),
            main_metric_name = score.metric_name,
            main_metric_score = score.score
        )
        self.track_primitive_executions(metadata, primitive_executions, task_execution)
        return task_execution


    def track_primitive_executions(self, metadata: dict, primitive_executions: list[BatchExecution], task_execution: TaskExecution):
        for p in primitive_executions:
            stats = TaskExecutionStats(p.batch.interactions, 0, 0)
            primitive_execution = self.tracker.create_primitive_execution(
                task_execution_id=task_execution.id,
                primitive_type=p.primitive.__class__.__name__,
                primitive_id=p.primitive.id,
                output_type=p.primitive.output_type.name,
                duration=p.duration_seconds or 0.0,
                num_inferences=len(p.batch),
                mean_inference_latency = stats.inference_latencies.mean(),
                median_inference_latency = stats.inference_latencies.median(),
                num_cached_inferences = stats.cached_inference_latencies.count(),
                num_cache_reads = stats.cache_read_latencies.count(),
                mean_cache_read_latency = stats.cache_read_latencies.mean(),
                median_cache_read_latency = stats.cache_read_latencies.median(),
                num_cache_writes = stats.cache_write_latencies.count(),
                mean_cache_write_latency = stats.cache_write_latencies.mean(),
                median_cache_write_latency = stats.cache_write_latencies.median(),
                mean_prompt_length = stats.prompt_lengths.mean(),
                mean_prediction_length = stats.output_lengths.mean(),
                output_text = str(p.batch.non_empty()),
                output_location = ''
            )
            if self.inference_tracking != TrackingMode.NONE:
                interactions = p.batch.interactions
                if self.inference_tracking == TrackingMode.RANDOM_SAMPLE_SINGLE:
                    interactions = [interactions[random.randint(0, len(interactions) - 1)]]
                elif self.inference_tracking == TrackingMode.RANDOM_SAMPLE_DOZEN:
                    interactions = random.sample(interactions, min(12, len(interactions)))
                else:
                    log(f'Evaluator is saving {len(interactions)} inferences to Spanner')
                self.track_inferences(metadata, interactions, task_execution, primitive_execution)

    def track_inferences(self, metadata: dict, interactions: list[Interaction], task_execution: TaskExecution, primitive_execution: PrimitiveExecution):
        for i in interactions:
            model_name, model_version = i.selected_model.parse_model_name_version() if i.selected_model else ('', '')
            self.tracker.create_inference(
                task_execution_id=task_execution.id,
                primitive_execution_id=primitive_execution.id,
                model_name=model_name,
                model_version=model_version,
                duration = i.timers.inference_latency.seconds,
                is_cached = i.flags.cache_hit,
                is_multimodal = not i.is_output_empty() and not i.output.is_text(),
                input_text = i.prompt.to_text(),
                input_location = metadata['input_location'] if 'input_location' in metadata else '',
                num_input_tokens = i.inference_metadata.num_input_tokens,
                output_text = i.raw_model_output_text or i.output.to_text(),
                output_location = metadata['output_location'] if 'output_location' in metadata else '',
                num_output_tokens = i.inference_metadata.num_output_tokens
            )

    def display_eval_summary(self, agg: TaskExecutionStats, eval_name: str, eval_version: str, num_of_tasks: int, sample_size: int, dataset_length: int, duration: float, inference_stats: str = '', mean_prompt_length_chars: float = 0.0):
        model = self.engine.default_model()
        log('='*50)
        log('Summary:')
        log(f' - Eval name: {eval_name}')
        log(f' - Eval version: {eval_version}')
        log(f' - Model name: {model.model_name}')
        log(f' - {model.model_config}')
        log(f' - Number of tasks: {num_of_tasks}')
        log(f' - Number of inputs: {sample_size if sample_size else dataset_length} (avg={mean_prompt_length_chars} chars)')
        log(f' - Number of inferences: {int(agg.num_inferences.sum())}')
        if self.trace > Trace.OFF:
            log(f' - Inference Stats: {inference_stats}')
        log(f' - Mean score: {agg.scores.mean()*100:.2f}%')
        log(f' - Median score: {agg.scores.median()*100:.2f}%')
        log(f' - Duration: {duration:.1f} seconds')
        log('='*50)
