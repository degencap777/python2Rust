import dataclasses
from enum import Enum
from typing import Type, Any
from google.cloud import spanner
from google.cloud.spanner_v1.transaction import Transaction
from utils.numerals import NumberUtils
from utils.strings import StringUtils
from utils.exceptions import InvalidArgsError
from utils.logger import log

# Conventions:
class ORM():
    def __init__(self, project_id: str, spanner_instance_id: str, spanner_database_id: str, data_classes: list[Type], ignore_class_name_prefixes: list[str] = []):
        """Handle Database persistency for simple tables by mapping DB records to data classes.
        - All dataclasses must have fields 'id' and 'insert_date'
        - New objects must have 'id' = None (this triggers insert instead of update)
        - All tables must have a plural name while related classes are singlular (EvalRun -> EvalRuns)
        - If data class name includes a prefix not found in the table name, add it to ignore_class_name_prefixes
        """
        self.client = spanner.Client(project_id)
        self.instance = self.client.instance(spanner_instance_id)
        self.database = self.instance.database(spanner_database_id)
        self.map_tables_to_classes = {}
        for data_class in data_classes:
            table_name = StringUtils.remove_prefixes(data_class.__name__, ignore_class_name_prefixes)
            self.map_tables_to_classes[table_name] = data_class
        log(f'ORM uses {project_id}/{spanner_instance_id}/{spanner_database_id}')


    def write(self, table_name: str, obj: Any, transaction: Transaction | None = None):
        """Insert or Update one record with or without transaction"""
        is_insert = False
        if not hasattr(obj, 'id') or not obj.id:
            obj.id = StringUtils.generate_database_id()
            is_insert = True
        if not hasattr(obj, 'insert_date') or not obj.insert_date:
            obj.insert_date = spanner.COMMIT_TIMESTAMP
            is_insert = True
        columns = []
        values = []
        for field in dataclasses.fields( obj ):
            columns.append(field.name)
            value = getattr(obj, field.name)
            if isinstance(value, Enum):
                value = str(value)
            if isinstance(value, float):
                value = NumberUtils.round_float( value )  # Yep that's exactly how opinionated we are.
            values.append(value)
        if transaction:
            if is_insert:
                transaction.insert(table_name, columns, [values])
            else:
                transaction.update(table_name, columns, [values])
        else:
            with self.database.batch() as batch:
                if is_insert:
                    batch.insert(table_name, columns, [values])
                else:
                    batch.update(table_name, columns, [values])

    def _get_dataclass_for_table(self, table_name: str):
        if not table_name:
            raise InvalidArgsError('id and table_name are required')
        class_name = table_name.rstrip('s')
        if not class_name in self.map_tables_to_classes: raise InvalidArgsError(f'table_name {table_name} is not registered.')
        return self.map_tables_to_classes[class_name]

    def _get_columns(self, data_class) -> list[str]:
        return [field.name for field in dataclasses.fields( data_class )]

    def _data_class_from_row(self, row, column_names: list[str], data_class):
        field_value_dict = {column_names[i]: row[i] for i in range(len(row))}
        return data_class(**field_value_dict)

    def read(self, table_name: str, id: str):
        if not id:
            raise InvalidArgsError('id and table_name are required')
        data_class = self._get_dataclass_for_table(table_name)
        column_names = self._get_columns(data_class)
        keyset = spanner.KeySet(keys = [[id]])
        with self.database.snapshot() as snapshot:
            rows = list(snapshot.read(table_name, columns=column_names, keyset=keyset))
            if len(rows) != 1:
                raise InvalidArgsError(f'{table_name} with id={id} does not exist.')
            row = rows[0]
            return self._data_class_from_row(row, column_names, data_class)

    def select_latest(self, table_name: str, page_size: int = 100, page_index: int = 0) -> list:
        if page_size < 1:
            raise InvalidArgsError('page_size must be a positive number')
        if page_index < 0:
            raise InvalidArgsError('page_index must not be negative')
        data_class = self._get_dataclass_for_table(table_name)
        column_names = self._get_columns(data_class)
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(f"""
                SELECT {', '.join(column_names)}
                FROM {table_name}
                ORDER BY insert_date DESC LIMIT @limit OFFSET @offset;
                """,
                params={'limit': page_size, 'offset': page_index * page_size},
                param_types={'limit': spanner.param_types.INT64, 'offset': spanner.param_types.INT64}
            )
            objects = []
            for row in results:
                objects.append(self._data_class_from_row(row, column_names, data_class))
            return objects

    def select_strings(self, sql: str, **kwargs: Any) -> list[str]:
        """Run any query that returns a list of strings, with optional named parameters"""
        if not sql:
            raise InvalidArgsError(f'select_strings called with an empty SQL')
        with self.database.snapshot() as snapshot:
            if kwargs:
                params = {}
                param_types = {}
                for key, value in kwargs.items():
                    if not isinstance(key, str):
                        raise InvalidArgsError(f'select_strings requires all parameter names to be strings')
                    params[key] = value
                    if isinstance(value, str):
                        param_types[key] = spanner.param_types.STRING
                    elif isinstance(value, int):
                        param_types[key] = spanner.param_types.INT64
                    elif isinstance(value, float):
                        param_types[key] = spanner.param_types.FLOAT64
                    elif isinstance(value, bool):
                        param_types[key] = spanner.param_types.BOOL
                    else:
                        raise InvalidArgsError(f'select_strings supports only string and integer parameters.')
                results = snapshot.execute_sql(sql, params=params, param_types=param_types)
            else:
                results = snapshot.execute_sql(sql)
            return [str(row[0]) for row in results]

    def select_int(self, sql: str, **kwargs: Any) -> int:
        """Run any query that returns a single integer, with optional named parameters"""
        if not sql:
            raise InvalidArgsError(f'select_int called with an empty SQL')
        with self.database.snapshot() as snapshot:
            if kwargs:
                params = {}
                param_types = {}
                for key, value in kwargs.items():
                    if not isinstance(key, str):
                        raise InvalidArgsError(f'select_int requires all parameter names to be strings')
                    params[key] = value
                    if isinstance(value, str):
                        param_types[key] = spanner.param_types.STRING
                    elif isinstance(value, int):
                        param_types[key] = spanner.param_types.INT64
                    elif isinstance(value, float):
                        param_types[key] = spanner.param_types.FLOAT64
                    elif isinstance(value, bool):
                        param_types[key] = spanner.param_types.BOOL
                    else:
                        raise InvalidArgsError(f'select_int supports only string and integer parameters.')
                results = snapshot.execute_sql(sql, params=params, param_types=param_types)
            else:
                results = snapshot.execute_sql(sql)
            int_results = [int(row[0]) for row in results]
            if len(int_results) != 1:
                raise InvalidArgsError(f'select_int called with a query that returned {len(int_results)} instead of one integer row.')
            return int_results[0]
