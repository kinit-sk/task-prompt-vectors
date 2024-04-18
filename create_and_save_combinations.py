import itertools
import operator

from arithmetics import TaskPrompt
from typing import List
from functools import reduce

import torch


origin_prompts = ["origin_0"]
dataset_names = ["mnli", "qnli"]

def get_task_prompts(origin_prompts, dataset_names, device = "cuda"):
    return {origin_prompt: [TaskPrompt(prompt_name, task_weights=torch.load(f"soft_prompts/{origin_prompt}/{prompt_name}.bin"), origin_weigts=torch.load(f"soft_prompts/{origin_prompt}/{origin_prompt}.bin")["prompt_embeddings"], device=device) for prompt_name in sorted(dataset_names)] for origin_prompt in origin_prompts}

def create_task_combinations(task_prompts : List[TaskPrompt], n : int = 2) -> List[TaskPrompt]:
    tp_cominations = itertools.combinations(sorted(task_prompts, key=lambda x: x.task_name), n)


    return [reduce(operator.add, tp) for tp in tp_cominations]

task_prompts_per_origin = get_task_prompts(origin_prompts, dataset_names)

for origin_prompt in task_prompts_per_origin:
    for task_prompt in create_task_combinations(task_prompts_per_origin[origin_prompt]):
        print(task_prompt.task_name)
        print(task_prompt.prompt)

        prompt = task_prompt.apply(torch.load(f"soft_prompts/{origin_prompt}/{origin_prompt}.bin")["prompt_embeddings"].to("cuda"))
        torch.save(prompt, f"soft_prompts/{origin_prompt}/{'_'.join(task_prompt.task_name.replace('+ ', '').split(' '))}.bin")
