from peft import PeftConfig

from dataclasses import field, dataclass


@dataclass
class PromptArithmeticsConfig:
    num_virtual_tokens: int = field(
        default=50,
        metadata={
            "help": "Size of the prompt (the actual prompt size is doubled in encoder-decoder models)."
        },
    )

    origin_prompt: str = field(
        default=None, metadata={"help": "Path to the origin prompt"}
    )
