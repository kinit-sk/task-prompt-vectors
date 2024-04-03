from dataclasses import field, dataclass
from typing import List

from peft import PromptTuningConfig


@dataclass
class PromptArithmeticsConfig(PromptTuningConfig):
    origin_prompts: List[str] = field(
        default=None, metadata={"help": "Path to the origin prompts"}
    )
