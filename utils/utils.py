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


def get_task_prompt_vectors_from_prompts(
    origin_prompts, dataset_names: List[str], device: str = "cuda"
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
        for origin_prompt in origin_prompts
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


def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)
