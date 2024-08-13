from dataclasses import dataclass, field
from core.data import Data
from utils.strings import StringUtils

@dataclass
class FewShotExample():
    input: Data
    output: Data
    source: str = ''  # optional, identifies the file where the example came from
    scores_by_index: dict[int, float] = field(default_factory=dict)  # temporary storage for last evalution scores
    score_contribution: float = 0.0  # temporary storage for percentage contribution to overall eval score

    def __repr__(self) -> str:
        source = f'[{self.source}] ' if self.source else ''
        input = StringUtils.truncate(self.input.value.strip(), 500, no_linebreaks=True)
        output = StringUtils.truncate(self.output.value.strip(), 500, no_linebreaks=True)
        return f'{source}{input} -> {output}'