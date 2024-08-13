from google.cloud import spanner
from config.defaults import Default
from utils.logger import log

class EvalSchema():
    def __init__(self, project_id: str, spanner_instance_id: str, spanner_database_id: str):
        self.project_id = project_id
        self.spanner_instance_id = spanner_instance_id
        self.spanner_database_id = spanner_database_id
        self.client = spanner.Client(project_id)
        self.instance = self.client.instance(spanner_instance_id)
        self.database = self.instance.database(spanner_database_id)

    def create_database(self):
        """Creates a new Spanner database for Evaluations."""
        self.database = self.instance.database(self.spanner_database_id)
        operation = self.database.create()
        log(f'Adding database {self.spanner_database_id} to Spanner instance {self.spanner_instance_id} in project {self.project_id}')
        operation.result(120)
        log('Evaluation DB is ready.')

    def create_table_eval_runs(self):
        log('Creating table EvalRuns...')
        operation = self.database.update_ddl(["""
        CREATE TABLE EvalRuns (
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_task_runs INT64,
            mean_score FLOAT64,
            median_score FLOAT64,
            num_inferences INT64,
            num_input_tokens INT64,
            num_output_tokens INT64,
            num_cached_inferences INT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('EvalRuns table created.')

    def create_table_task_runs(self):
        log('Creating table TaskRuns...')
        operation = self.database.update_ddl(["""
        CREATE TABLE TaskRuns (
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_executions INT64,
            mean_execution_latency FLOAT64,
            median_execution_latency FLOAT64,
            min_execution_latency FLOAT64,
            max_execution_latency FLOAT64,
            num_inferences INT64,
            mean_inference_latency FLOAT64,
            median_inference_latency FLOAT64,
            num_cached_inferences INT64,
            num_cache_reads INT64,
            mean_cache_read_latency FLOAT64,
            median_cache_read_latency FLOAT64,
            num_cache_writes INT64,
            mean_cache_write_latency FLOAT64,
            median_cache_write_latency FLOAT64,
            mean_prompt_length FLOAT64,
            mean_prediction_length FLOAT64,
            mean_score FLOAT64,
            median_score FLOAT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('TaskRuns table created.')

    def create_table_task_executions(self):
        log('Creating table TaskExecutions...')
        operation = self.database.update_ddl(["""
        CREATE TABLE TaskExecutions(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            task_run_id STRING(MAX) NOT NULL,
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_inferences INT64,
            mean_inference_latency FLOAT64,
            median_inference_latency FLOAT64,
            num_cached_inferences INT64,
            num_cache_reads INT64,
            mean_cache_read_latency FLOAT64,
            median_cache_read_latency FLOAT64,
            num_cache_writes INT64,
            mean_cache_write_latency FLOAT64,
            median_cache_write_latency FLOAT64,
            mean_prompt_length FLOAT64,
            mean_prediction_length FLOAT64,
            input_text STRING(MAX),
            input_location STRING(MAX),
            output_text STRING(MAX),
            output_location STRING(MAX),
            output_context STRING(MAX),
            last_inference_output STRING(MAX),
            ground_truth STRING(MAX),
            main_metric_name STRING(MAX),
            main_metric_score FLOAT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('TaskExecutions table created.')

    def create_table_primitive_executions(self):
        log('Creating table PrimitiveExecutions...')
        operation = self.database.update_ddl(["""
        CREATE TABLE PrimitiveExecutions(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            primitive_type STRING(MAX) NOT NULL,
            primitive_id STRING(MAX) NOT NULL,
            output_type STRING(MAX) NOT NULL,
            eval_name STRING(MAX) NOT NULL,
            eval_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            task_run_id STRING(MAX) NOT NULL,
            task_execution_id STRING(MAX) NOT NULL,
            model_name STRING(MAX),
            model_version STRING(MAX),
            dataset_location STRING(MAX),
            dataset_version STRING(MAX),
            duration FLOAT64,
            num_inferences INT64,
            mean_inference_latency FLOAT64,
            median_inference_latency FLOAT64,
            num_cached_inferences INT64,
            num_cache_reads INT64,
            mean_cache_read_latency FLOAT64,
            median_cache_read_latency FLOAT64,
            num_cache_writes INT64,
            mean_cache_write_latency FLOAT64,
            median_cache_write_latency FLOAT64,
            mean_prompt_length FLOAT64,
            mean_prediction_length FLOAT64,
            input_text STRING(MAX),
            input_location STRING(MAX),
            output_text STRING(MAX),
            output_location STRING(MAX),
            output_context STRING(MAX),
            ground_truth STRING(MAX),
            main_metric_name STRING(MAX),
            main_metric_score FLOAT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('PrimitiveExecutions table created.')

    def create_table_inferences(self):
        log('Creating table Inferences...')
        operation = self.database.update_ddl(["""
        CREATE TABLE Inferences(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            eval_run_id STRING(MAX) NOT NULL,
            task_run_id STRING(MAX) NOT NULL,
            task_execution_id STRING(MAX) NOT NULL,
            primitive_execution_id STRING(MAX) NOT NULL,
            model_name STRING(MAX) NOT NULL,
            model_version STRING(MAX),
            duration FLOAT64,
            is_cached BOOL NOT NULL,
            is_multimodal BOOL NOT NULL,
            input_text STRING(MAX),
            input_location STRING(MAX),
            num_input_tokens INT64,
            output_text STRING(MAX),
            output_location STRING(MAX),
            num_output_tokens INT64
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Inferences table created.')

    def create_table_scores(self):
        log('Creating table Scores...')
        operation = self.database.update_ddl(["""
        CREATE TABLE TaskScores(
            id STRING(MAX) NOT NULL,
            namespace STRING(MAX) NOT NULL,
            insert_date TIMESTAMP NOT NULL OPTIONS(allow_commit_timestamp=true),
            task_execution_id STRING(MAX) NOT NULL,
            task_run_id STRING(MAX) NOT NULL,
            task_name STRING(MAX) NOT NULL,
            task_version STRING(MAX),
            eval_run_id STRING(MAX) NOT NULL,
            rater_name STRING(MAX) NOT NULL,
            rater_version STRING(MAX),
            metric_category STRING(MAX),
            metric_name STRING(MAX) NOT NULL,
            is_main_metric BOOL NOT NULL,
            score FLOAT64 NOT NULL
        )
        PRIMARY KEY (id)
        """])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Scores table created.')

    def create_indexes(self):
        log('Creating Spanner indexes...')
        operation = self.database.update_ddl([
            'CREATE UNIQUE INDEX EvalRuns_NamespaceNameVersion ON EvalRuns (namespace, eval_name, eval_version)',
            'CREATE INDEX TaskRuns_EvalRunId ON TaskRuns (eval_run_id)',
            'CREATE INDEX TaskExecutions_EvalRunId ON TaskExecutions (eval_run_id)',
            'CREATE INDEX TaskExecutions_TaskRunId ON TaskExecutions (task_run_id)',
            'CREATE INDEX PrimitiveExecutions_TaskExecutionId ON PrimitiveExecutions (task_execution_id)',
            'CREATE INDEX Inferences_PrimitiveExecutionId ON Inferences (primitive_execution_id)',
            'CREATE INDEX TaskScores_TaskExecutionId ON TaskScores (task_execution_id)'
        ])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Indexes are ready.')

    def create_schema(self):
        self.create_table_eval_runs()
        self.create_table_task_runs()
        self.create_table_task_executions()
        self.create_table_primitive_executions()
        self.create_table_inferences()
        self.create_table_scores()
        self.create_indexes()
        log('Successfully created all tables and indexes.')

    def drop_schema(self):
        log('Deleting all Spanner indexes and tables for Evaluations')
        operation = self.database.update_ddl([
            # 'DROP INDEX EvalRuns_NamespaceNameVersion',
            # 'DROP INDEX TaskExecutions_EvalRunId',
            # 'DROP INDEX TaskExecutions_TaskRunId',
            # 'DROP INDEX TaskRuns_EvalRunId',
            # 'DROP INDEX TaskScores_TaskExecutionId',
            # 'DROP INDEX PrimitiveExecutions_TaskExecutionId',
            # 'DROP INDEX Inferences_PrimitiveExecutionId',
            # 'DROP TABLE EvalRuns',
            # 'DROP TABLE TaskRuns',
            # 'DROP TABLE TaskExecutions',
            # 'DROP TABLE PrimitiveExecutions',
            # 'DROP TABLE Inferences',
            # 'DROP TABLE TaskScores'
        ])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Deleted all tables and indexes.')

    def add_columns(self):
        log('Adding new columns...')
        operation = self.database.update_ddl([
            'ALTER TABLE TaskExecutions ADD COLUMN last_inference_output STRING(MAX)',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_inferences INT64',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_input_tokens INT64',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_output_tokens INT64',
            # 'ALTER TABLE EvalRuns ADD COLUMN num_cached_inferences INT64'
        ])
        operation.result(Default.SPANNER_TIMEOUT)
        log('Finished adding columns.')
