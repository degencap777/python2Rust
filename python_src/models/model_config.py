import hashlib
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    generation: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    grounding_source: object = None

    def __repr__(self) -> str:
        params = []
        for k, v in self.generation.items():
            params.append(f'{k}={v}')
        if self.grounding_source:
            params.append(f"grounding_source=ON")
        return f"ModelConfig: {', '.join(params)}"

    def hash_generation_config(self) -> str:
        """Hash of the generation config serves as a part of the cache key, so that model outpus for each combination of generation parameters are cached separately."""
        DIGEST_LENGTH = 6  # Keeping it short because inference cache is expected to have millions of keys
        if not self.generation:
            return ''
        kev_value_pairs = []
        for k, v in self.generation.items():
            kev_value_pairs.append(f'{k}={v}')
        csv = ','.join(kev_value_pairs)
        return hashlib.sha256(csv.encode('utf-8')).hexdigest()[:DIGEST_LENGTH]