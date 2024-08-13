from google.cloud import spanner

from core.orm import ORM

from eval.eval_run import EvalRun
from eval.task_run import TaskRun
from eval.task_execution import TaskExecution
from eval.task_score import TaskScore
from eval.primitive_execution import PrimitiveExecution
from eval.inference import Inference

from utils.logger import log
from utils.exceptions import InvalidArgsError


class EvalResults():
    """Read only API for fetching evaluation results from Spanner."""

    def __init__(self, project_id: str, spanner_instance_id: str, spanner_database_id: str):
        dataclasses = [EvalRun, TaskRun, TaskExecution, TaskScore, PrimitiveExecution, Inference]
        self.orm = ORM(project_id, spanner_instance_id, spanner_database_id, dataclasses)
        spanner_instance = spanner.Client(project_id).instance(spanner_instance_id)
        self.database = spanner_instance.database(spanner_database_id)

    def load_task_executions(self, task_run_id: str | None = None, eval_run_id: str | None = None) -> list[TaskExecution]:
        if not task_run_id and not eval_run_id:
            raise InvalidArgsError('Either task_run_id or task_run_ids must be specified')
        if task_run_id and eval_run_id:
            raise InvalidArgsError('Only one of task_run_id or task_run_ids can be specified')
        with self.database.snapshot() as snapshot:
            if task_run_id:
                sql = 'SELECT id FROM TaskExecutions WHERE task_run_id = @param'
            else:
                sql = 'SELECT id FROM TaskExecutions WHERE eval_run_id = @param'
            log(f'Loading task executions...')
            cursor = snapshot.execute_sql(
                sql,
                params={'param': task_run_id or eval_run_id},
                param_types={'param': spanner.param_types.STRING}
            )
            results = []
            for row in cursor:
                results.append(self.orm.read('TaskExecutions', row[0]))
            log(f'Loaded {len(results)} task executions')
            return results

    def load_inferences(self, task_execution_id: str) -> list[Inference]:
        if not task_execution_id:
            raise InvalidArgsError('task_execution_id must be specified')
        with self.database.snapshot() as snapshot:
            cursor = snapshot.execute_sql(
                'SELECT id FROM Inferences WHERE task_execution_id = @param',
                params={'param': task_execution_id},
                param_types={'param': spanner.param_types.STRING}
            )
            results = []
            for row in cursor:
                results.append(self.orm.read('Inferences', row[0]))
            log(f'Loaded {len(results)} inferences for task execution {task_execution_id}')
            return results
