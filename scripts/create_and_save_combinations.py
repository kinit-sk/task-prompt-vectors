import itertools
import operator

from arithmetics import TaskPrompt
from typing import List
from functools import reduce

import torch

import numpy as np


# origin_prompts = [
#     "origin_0",
#     "origin_1",
#     "origin_2",
# ]

origin_prompts = [
    "origin_0_meta-llama-3.1-8b-instruct",
    "origin_1_meta-llama-3.1-8b-instruct",
    "origin_2_meta-llama-3.1-8b-instruct",
]

# NLI tasks
# dataset_names = ["mnli_text", "qnli_text"]

# classificaiton tasks
# dataset_names = ["dbpedia_text", "trec_coarse_text"]

# sentiment tasks
# dataset_names = ["sst2_text", "yelp_polarity_text"]

# dataset_names = [
#     "mnli_text",
#     "qnli_text",
#     "dbpedia_text",
#     "trec_coarse_text",
#     "sst2_text",
#     "yelp_polarity_text",
# ]

# dataset_names = [
#     "mnli_text_instruct",
#     "qnli_text_instruct",
#     "dbpedia_text_instruct",
#     "trec_coarse_text_instruct",
#     "sst2_text_instruct",
#     "yelp_polarity_text_instruct",
# ]

dataset_names = [
    "rte_text_instruct",
    "stsb_text_instruct",
    "mrpc_text_instruct",
    "cola_text_instruct",
    "qqp_text_instruct",
]


def get_task_prompts(origin_prompts, dataset_names, device="cuda"):
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


def create_task_combinations(
    task_prompts: List[TaskPrompt], n: int = 2
) -> List[TaskPrompt]:
    tp_cominations = itertools.combinations(
        sorted(task_prompts, key=lambda x: x.task_name), n
    )

    return [reduce(operator.add, tp) for tp in tp_cominations]


def average_task_prompts(task_prompts_dict):
    # Extract the list of origins
    origins = list(task_prompts_dict.keys())

    # Initialize the array for average vectors
    avg_tpvs = {}

    # Assuming all origins have the same number of task prompts
    num_task_prompts = len(task_prompts_dict[origins[0]])

    # Loop through each task prompt index
    for i in range(num_task_prompts):
        # Collect all vectors for the current task prompt index from each origin
        tpvs = [task_prompts_dict[origin][i] for origin in origins]

        # Compute the average vector
        avg_tpv = reduce(operator.add, tpvs)
        avg_tpv.prompt /= num_task_prompts

        # Append the average vector to the result list
        avg_tpvs[list(avg_tpv.tasks)[0]] = avg_tpv

    return avg_tpvs


task_prompts_per_origin = get_task_prompts(origin_prompts, dataset_names)

for origin_prompt in task_prompts_per_origin:
    for task_prompt in create_task_combinations(task_prompts_per_origin[origin_prompt]):
        print(task_prompt.task_name)
        print(task_prompt.prompt)

        prompt = task_prompt.apply(
            torch.load(f"soft_prompts/{origin_prompt}/{origin_prompt}.bin")[
                "prompt_embeddings"
            ].to("cuda")
        )
        torch.save(
            prompt,
            f"soft_prompts/{origin_prompt}/{'_'.join(task_prompt.task_name.replace('+ ', '').split(' '))}.bin",
        )

# avg_tpvs = average_task_prompts(task_prompts_per_origin)
# for origin_prompt in origin_prompts:
#     for task in avg_tpvs:
#         print(origin_prompt, avg_tpvs[task].task_name)

#         prompt = avg_tpvs[task].apply(
#             torch.load(f"soft_prompts/{origin_prompt}/{origin_prompt}.bin")[
#                 "prompt_embeddings"
#             ].to("cuda")
#         )

#         torch.save(
#             prompt,
#             f"soft_prompts/{origin_prompt}/{task}_avg_10.bin",
#         )
