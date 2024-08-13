from dataclasses import dataclass

@dataclass
class InferenceMetadata():
    num_input_tokens: int = 0
    num_output_tokens: int = 0

    def __repr__(self) -> str:
        return f'num_input_tokens={self.num_input_tokens}, num_output_tokens={self.num_output_tokens}'
