import torch

from typing import List, Dict

from arithmetics import TaskPrompt, PromptArithmeticsConfig

from safetensors.torch import safe_open


def get_task_prompt_vectors(
    pa_config: PromptArithmeticsConfig, dataset_names: List[str], device: str = "cuda"
) -> Dict[str, List[TaskPrompt]]:
    return {
        origin_prompt: [
            TaskPrompt(
                prompt_name,
                task_weights=torch.load(
                    f"soft_prompts/{origin_prompt}/{prompt_name}.bin"
                ),
                origin_weigts=torch.load(
                    f"soft_prompts/{origin_prompt}/{origin_prompt}.bin"
                )["prompt_embeddings"],
                device=device,
            )
            for prompt_name in sorted(dataset_names)
        ]
        for origin_prompt in pa_config.origin_prompts
    }


def get_task_prompts(
    pa_config: PromptArithmeticsConfig, dataset_names: List[str], device: str = "cuda"
) -> Dict[str, List[torch.Tensor]]:
    return {
        origin_prompt: [
            torch.load(f"soft_prompts/{origin_prompt}/{prompt_name}.bin").to(device)
            for prompt_name in sorted(dataset_names)
        ]
        for origin_prompt in pa_config.origin_prompts
    }

def get_task_prompt_from_safetensor(save: str):
    tensors = {}

    with safe_open(f"{save}/adapter_model.safetensors", framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    return torch.nn.Parameter(tensors["prompt_embeddings"])

