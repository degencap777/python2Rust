from abc import ABCMeta, abstractmethod
from core.data_bundle import DataBundle
from core.interaction import Interaction
from core.metric_category import MetricCategory
from core.score import Score
from engine.engine import Engine
from utils.exceptions import AbstractMethodNotImplementedError
from utils.strings import StringUtils


class Task(object, metaclass=ABCMeta):
    """Abstract base class for Tasks that process one input at a time."""

    force_model = None  # Optional, task implementations can use it to store a hardcoded model.

    @property
    def task_id(self) -> str:
        """Derive default task id from class name. User can override this property by setting class attribute id."""
        if hasattr(self, 'id') and self.id and isinstance(self.id, str):  # type: ignore
            return self.id  # type: ignore
        return StringUtils.camel_to_snake(self.__class__.__name__)

    @property
    def task_version(self) -> str:
        """If task versioning is needed, custom tasks must override this property."""
        return 'V1.0'

    @abstractmethod
    def run(self, engine: Engine, context: Interaction) -> DataBundle:
        raise AbstractMethodNotImplementedError()

    def rate(self, task_result: DataBundle, ground_truth: DataBundle) -> Score | list[Score]:
        """Default implemenatation uses ExactMatch, which users can override."""
        if task_result.is_text() and ground_truth.is_text(): # Case insensitive comparison for strings
            is_exact_match = (task_result.to_text().strip().lower() == ground_truth.to_text().strip().lower())
        else:
            is_exact_match = (task_result == ground_truth)
        score = 1.0 if is_exact_match else 0.0
        # log(f'Task.rate returns {score} for {task_result} and {ground_truth}')
        return Score(MetricCategory.ACCURACY, 'exact_match', score)

    def citation_instruction(self, result: DataBundle) -> str | None:
        """Generates an instruction line for finding a tag like '<SECTION-123>' that contains information most relevant to the given task result.

        The main benefit of defining citation instructions in each task are:
        - The citation instruction is managed in the same file with the main task.
        - 'task_id' property can be used to combine multiple citation prompts in a single QuestionSet primitive.

        Args:
        - task_run_result: DataBundle with the result of task execution. This can be embedded into the citation instruction in order to help the model find the right <SECTION>.
        """
        return None

    def citation_claim(self, result: DataBundle) -> str | None:
        """Generates a claim that can be tested against individual pages or text chunks using an LLM or API

        The main benefit of defining citation claims in each task are:
        - The citation claim is managed in the same file with the main task.
        - 'task_id' property can be used to combine multiple citation prompts in a single QuestionSet primitive.

        Args:
        - task_run_result: DataBundle with the result of task execution. This can be embedded into the citation instruction in order to help the model find the right <SECTION>.
        """
        return None

    def relevancy_filters(self) -> list[str]:
        """Returns a list of definitions for filtering relevant content"""

        return []

    def has_find_relevant_text(self) -> bool:
        """Must return True whenever the task provides the find_relevant_text method."""

        return False

    def find_relevant_text(self, engine: Engine, document: str) -> str | None:
        """Returns a string with one or more extracts from the given document that are relevant to this task."""

        return None

    def __repr__(self) -> str:
        return f'{type(self)} "{self.task_id}"'