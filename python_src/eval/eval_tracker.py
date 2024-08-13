from eval.eval_run import EvalRun
from eval.inference import Inference
from eval.task_run import TaskRun
from eval.task_execution import TaskExecution
from eval.primitive_execution import PrimitiveExecution
from eval.task_score import TaskScore

from core.orm import ORM

from engine.scaler import Scaler

from utils.exceptions import InvalidArgsError, InvalidContext
from utils.strings import StringUtils
from utils.logger import Trace, log

class EvalTracker:
    def __init__(self, project_id: str, spanner_instance_id: str, database_id: str, namespace: str = 'DEFAULT', trace: Trace = Trace.OFF):
        dataclasses = [EvalRun, Inference, PrimitiveExecution, TaskRun, TaskExecution, TaskScore]
        self.orm = ORM(project_id, spanner_instance_id, database_id, dataclasses)
        self.namespace = namespace
        self.eval_run: EvalRun
        self.task_run: TaskRun
        self.trace = trace

    def start_eval_run(
        self,
        eval_name: str,
        eval_version: str,
        model_name: str,
        model_version: str,
        dataset_location: str,
        dataset_version: str,
    ):
        params = (eval_name, eval_version, model_name, model_version, dataset_location, dataset_version)
        if None in params:
            raise InvalidArgsError('All arguments are required.')
        if hasattr(self, 'eval_run') and self.eval_run:
            raise InvalidContext('EvaluationRun already started.')
        self.eval_run = EvalRun(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            eval_name=eval_name,
            eval_version=eval_version,
            model_name=model_name,
            model_version=model_version,
            dataset_location=dataset_location,
            dataset_version=dataset_version
        )
        self.orm.write('EvalRuns', self.eval_run)
        log(f'Started EvalRun #{self.eval_run.id} {self.eval_run.eval_name}:{self.eval_run.eval_version}')
        return self.eval_run

    def finalize_eval_run(self, scaler: Scaler, duration: float, num_task_runs: int, mean_score: float, median_score: float, inference_stats: str = ''):
        self.eval_run.duration = duration
        self.eval_run.num_task_runs = num_task_runs
        self.eval_run.mean_score = mean_score
        self.eval_run.median_score = median_score
        self.eval_run.num_inferences = scaler.num_inferences
        self.eval_run.num_cached_inferences = scaler.num_cached_inferences
        self.eval_run.num_input_tokens = scaler.num_input_tokens
        self.eval_run.num_output_tokens = scaler.num_output_tokens
        self.orm.write('EvalRuns', self.eval_run)
        log(f'EvalRun {self.eval_run.id} took {duration:.1f} seconds, score={mean_score*100:.2f}% {inference_stats}')
        return self.eval_run

    def start_task_run(self, task_name: str, task_version: str, dataset_location: str, dataset_version: str):
        if None in (task_name, task_version, dataset_location, dataset_version):
            raise InvalidArgsError('All arguments are required: task_name, task_version')
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before TaskRun')
        self.task_run = TaskRun(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            eval_name=self.eval_run.eval_name,
            eval_version=self.eval_run.eval_version,
            eval_run_id=self.eval_run.id,
            task_name=task_name,
            task_version=task_version,
            model_name=self.eval_run.model_name,
            model_version=self.eval_run.model_version,
            dataset_location=dataset_location,
            dataset_version=dataset_version,
        )
        self.orm.write('TaskRuns', self.task_run)
        log(f'Started TaskRun #{self.task_run.id} for {self.task_run.task_name}:{self.task_run.task_version}')
        return self.task_run

    def finalize_task_run(
        self,
        duration: float,
        num_executions: int,
        mean_execution_latency: float,
        median_execution_latency: float,
        min_execution_latency: float,
        max_execution_latency: float,
        num_inferences: int,
        mean_inference_latency: float,
        median_inference_latency: float,
        num_cached_inferences: int,
        num_cache_reads: int,
        mean_cache_read_latency: float,
        median_cache_read_latency: float,
        num_cache_writes: int,
        mean_cache_write_latency: float,
        median_cache_write_latency: float,
        mean_prompt_length: float,
        mean_prediction_length: float,
        mean_score: float,
        median_score: float,
    ):
        self.task_run.duration = duration
        self.task_run.num_executions = num_executions
        self.task_run.mean_execution_latency = mean_execution_latency
        self.task_run.median_execution_latency = median_execution_latency
        self.task_run.min_execution_latency = min_execution_latency
        self.task_run.max_execution_latency = max_execution_latency
        self.task_run.num_inferences = num_inferences
        self.task_run.mean_inference_latency = mean_inference_latency
        self.task_run.median_inference_latency = median_inference_latency
        self.task_run.num_cached_inferences = num_cached_inferences
        self.task_run.num_cache_reads = num_cache_reads
        self.task_run.mean_cache_read_latency = mean_cache_read_latency
        self.task_run.median_cache_read_latency = median_cache_read_latency
        self.task_run.num_cache_writes = num_cache_writes
        self.task_run.mean_cache_write_latency = mean_cache_write_latency
        self.task_run.median_cache_write_latency = median_cache_write_latency
        self.task_run.mean_prompt_length = mean_prompt_length
        self.task_run.mean_prediction_length = mean_prediction_length
        self.task_run.mean_score = mean_score
        self.task_run.median_score = median_score
        self.orm.write('TaskRuns', self.task_run)
        log(f'TaskRun {self.task_run.id} ({self.task_run.task_name}) took {duration:.1f} seconds, score={mean_score*100:.2f}%')
        return self.task_run

    def create_task_execution(
        self,
        duration: float,
        num_inferences: int,
        mean_inference_latency: float,
        median_inference_latency: float,
        num_cached_inferences: int,
        num_cache_reads: int,
        mean_cache_read_latency: float,
        median_cache_read_latency: float,
        num_cache_writes: int,
        mean_cache_write_latency: float,
        median_cache_write_latency: float,
        mean_prompt_length: float,
        mean_prediction_length: float,
        input_text: str,
        input_location: str,
        output_text: str,
        output_location: str,
        output_context: str,
        last_inference_output: str,
        ground_truth: str,
        main_metric_name: str,
        main_metric_score: float,
    ) -> TaskExecution:
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before TaskExecution')
        if not self.task_run:
            raise InvalidArgsError('TaskRun must be started before TaskExecution')
        execution = TaskExecution(
            id=StringUtils.generate_database_id(),
            eval_name=self.eval_run.eval_name,
            eval_version=self.eval_run.eval_version,
            eval_run_id=self.eval_run.id,
            task_name=self.task_run.task_name,
            task_version=self.task_run.task_version,
            task_run_id=self.task_run.id,
            model_name=self.task_run.model_name,
            model_version=self.task_run.model_version,
            dataset_location=self.task_run.dataset_location,
            dataset_version=self.task_run.dataset_version,
            duration=duration,
            num_inferences=num_inferences,
            mean_inference_latency=mean_inference_latency,
            median_inference_latency=median_inference_latency,
            num_cached_inferences=num_cached_inferences,
            num_cache_reads=num_cache_reads,
            mean_cache_read_latency=mean_cache_read_latency,
            median_cache_read_latency=median_cache_read_latency,
            num_cache_writes=num_cache_writes,
            mean_cache_write_latency=mean_cache_write_latency,
            median_cache_write_latency=median_cache_write_latency,
            mean_prompt_length=mean_prompt_length,
            mean_prediction_length=mean_prediction_length,
            input_text=input_text,
            input_location=input_location,
            output_text=output_text,
            output_location=output_location,
            output_context=output_context,
            last_inference_output = last_inference_output,
            ground_truth=ground_truth,
            main_metric_name=main_metric_name,
            main_metric_score=float(main_metric_score),
            namespace=self.namespace
        )
        self.orm.write('TaskExecutions', execution)
        return execution

    def create_task_score(
        self,
        task_execution_id: str,
        rater_name: str,
        rater_version: str,
        metric_category: str,
        metric_name: str,
        is_main_metric: bool,
        score: float,
    ) -> TaskScore:
        task_score = TaskScore(
            id=StringUtils.generate_database_id(),
            task_execution_id = task_execution_id,
            task_run_id=self.task_run.id,
            task_name=self.task_run.task_name,
            task_version=self.task_run.task_version,
            eval_run_id=self.eval_run.id,
            rater_name=rater_name,
            rater_version=rater_version,
            metric_category=metric_category,
            metric_name=metric_name,
            is_main_metric=is_main_metric,
            score=float(score),
            namespace=self.namespace
        )
        self.orm.write('TaskScores', task_score)
        return task_score

    def create_primitive_execution(
        self,
        task_execution_id: str,
        primitive_type: str,
        primitive_id: str,
        output_type: str,
        duration: float,
        num_inferences: int,
        mean_inference_latency: float,
        median_inference_latency: float,
        num_cached_inferences: int,
        num_cache_reads: int,
        mean_cache_read_latency: float,
        median_cache_read_latency: float,
        num_cache_writes: int,
        mean_cache_write_latency: float,
        median_cache_write_latency: float,
        mean_prompt_length: float,
        mean_prediction_length: float,
        output_text: str,
        output_location: str,
        input_text: str = '',
        input_location: str = ''
    ) -> PrimitiveExecution:
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before PrimitiveExecution')
        if not self.task_run:
            raise InvalidArgsError('TaskRun must be started before PrimitiveExecution')
        execution = PrimitiveExecution(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            primitive_type=primitive_type,
            primitive_id=primitive_id,
            output_type=output_type,
            eval_name=self.eval_run.eval_name,
            eval_version=self.eval_run.eval_version,
            eval_run_id=self.eval_run.id,
            task_name=self.task_run.task_name,
            task_version=self.task_run.task_version,
            task_run_id=self.task_run.id,
            task_execution_id=task_execution_id,
            model_name=self.task_run.model_name,
            model_version=self.task_run.model_version,
            dataset_location=self.task_run.dataset_location,
            dataset_version=self.task_run.dataset_version,
            duration=duration,
            num_inferences=num_inferences,
            mean_inference_latency=mean_inference_latency,
            median_inference_latency=median_inference_latency,
            num_cached_inferences=num_cached_inferences,
            num_cache_reads=num_cache_reads,
            mean_cache_read_latency=mean_cache_read_latency,
            median_cache_read_latency=median_cache_read_latency,
            num_cache_writes=num_cache_writes,
            mean_cache_write_latency=mean_cache_write_latency,
            median_cache_write_latency=median_cache_write_latency,
            mean_prompt_length=mean_prompt_length,
            mean_prediction_length=mean_prediction_length,
            input_text=input_text,
            input_location=input_location,
            output_text=output_text,
            output_location=output_location
        )
        self.orm.write('PrimitiveExecutions', execution)
        return execution

    def create_inference(
        self,
        task_execution_id: str,
        primitive_execution_id: str,
        model_name: str,
        model_version: str,
        duration: float,
        is_cached: bool,
        is_multimodal: bool,
        input_text: str,
        input_location: str,
        num_input_tokens: int,
        output_text: str,
        output_location: str,
        num_output_tokens: int
    ) -> Inference:
        if not self.eval_run:
            raise InvalidArgsError('EvalRun must be started before tracking Inferences')
        if not self.task_run:
            raise InvalidArgsError('TaskRun must be started before tracking Inferences')
        inference = Inference(
            id=StringUtils.generate_database_id(),
            namespace=self.namespace,
            eval_run_id = self.eval_run.id,
            task_run_id = self.task_run.id,
            task_execution_id = task_execution_id,
            primitive_execution_id = primitive_execution_id,
            model_name = model_name,
            model_version = model_version,
            duration = duration,
            is_cached = is_cached,
            is_multimodal = is_multimodal,
            input_text = input_text,
            input_location = input_location,
            num_input_tokens = num_input_tokens,
            output_text = output_text,
            output_location = output_location,
            num_output_tokens = num_output_tokens,
        )
        self.orm.write('Inferences', inference)
        return inference
