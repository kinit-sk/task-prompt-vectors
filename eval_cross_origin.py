from evaluator import ArithmeticsEvaluator
from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig, TaskPrompt
from tasks import Preprocessor
from utils import get_task_prompts, create_heatmaps

import torch


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import TaskType, PromptTuningConfig, get_peft_model

import os

import numpy as np

from datetime import datetime

from typing import Callable, Dict, List

from torch.nn.functional import cosine_similarity

from tqdm import tqdm


def average_diff(t1, t2):
    return torch.abs(t1 - t2).mean()

def l2_norm(t1,t2):
    return torch.abs(torch.norm(t1) - torch.norm(t2))

def cosine_sim(t1, t2):
    return cosine_similarity(t1.flatten(), t2.flatten(), dim=0)


def get_comparison(data_args: DataTrainingArguments, task_prompt_vectors: Dict[str, List[TaskPrompt]], function: Callable):
    cross_origin_comparisons = {}
    for i in tqdm(range(len(data_args.dataset_names))):
        # print(data_args.dataset_names[i])
        cross_origin_comparisons[data_args.dataset_names[i]] = []
        for o1 in task_prompt_vectors:
            cross_origin_comparisons[data_args.dataset_names[i]].append([])
            for o2 in task_prompt_vectors:
                # print(o1, o2)
                tpv1 = task_prompt_vectors[o1][i]
                tpv2 = task_prompt_vectors[o2][i]

                cross_origin_comparisons[data_args.dataset_names[i]][-1].append(function(tpv1.prompt, tpv2.prompt).item())

    return cross_origin_comparisons

timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pa_config = parser.parse_toml_file("configs/addition.toml")

task_prompt_vectors = get_task_prompts(pa_config=pa_config, dataset_names=data_args.dataset_names)

name_func_map = {"diff": average_diff, "l2": l2_norm, "cosine": cosine_sim}

for fname in name_func_map:
    print(fname)
    cross_origin_comparisons = get_comparison(data_args, task_prompt_vectors, name_func_map[fname])
    create_heatmaps(cross_origin_comparisons, filename_prefix=f"{fname}_{timestamp}")


exit()

os.environ["WANDB_PROJECT"] = training_args.wandb_project


tokenizer = AutoTokenizer.from_pretrained(
    data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_name_or_path)

peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=pa_config.num_virtual_tokens,
)

model = get_peft_model(model, peft_config)
model.base_model.generation_config.max_new_tokens = data_args.max_target_length


preprocessor = Preprocessor(data_args.dataset_names, data_args, training_args)

_, valid_datasets, test_datasets = preprocessor.get_data()

