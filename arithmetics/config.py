from dataclasses import field, dataclass
from typing import List

from peft import PromptTuningConfig


@dataclass
class PromptArithmeticsConfig(PromptTuningConfig):
    origin_prompts: List[str] = field(
        default=None, metadata={"help": "Origin prompt names."}
    )

    init_prompts: List[str] = field(
        default=None,
        metadata={"help": "Init prompt names to initialize prompt tuning."},
    )
