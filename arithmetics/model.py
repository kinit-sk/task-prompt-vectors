from peft import PeftModel, PromptTuningConfig, get_peft_model, PeftType, TaskType
from typing import Any

import torch

from .config import PromptArithmeticsConfig
from .task_prompt import TaskPrompt


class PromptArithmeticsModel:
    def __init__(self, peft_model: PeftModel, pa_config: PromptArithmeticsConfig):
        self.pa_config = pa_config
        self.peft_model = peft_model

        self.origin_prompt = torch.load(self.pa_config.origin_prompt)[
            "prompt_embeddings"
        ]

        self.peft_model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            self.origin_prompt
        )

    def set_task(self, task_prompt: TaskPrompt):
        self.peft_model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            self.origin_prompt + task_prompt.prompt
        )

    def set_origin(self):
        self.peft_model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            self.origin_prompt
        )


def get_pa_model(
    model: PeftModel, pa_config: PromptArithmeticsConfig
) -> PromptArithmeticsModel:
    peft_config = PromptTuningConfig(
        peft_type=PeftType.PROMPT_TUNING,
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=pa_config.num_virtual_tokens,
    )

    peft_model = get_peft_model(model, peft_config)

    return PromptArithmeticsModel(peft_model, pa_config)
