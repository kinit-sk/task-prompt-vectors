from evaluator import ArithmeticsEvaluator
from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig, TaskPrompt
from tasks import Preprocessor

import torch
import itertools
import operator

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
from functools import reduce
from peft import TaskType, PromptTuningConfig, get_peft_model

import os

import pandas as pd

# import transformers
# transformers.logging.set_verbosity_debug()

# create set of task prompts per origin prompt (for stability purposes)
def get_task_prompts(pa_config, dataset_names):
    return {origin_prompt: [TaskPrompt(prompt_name, task_weights=torch.load(f"soft_prompts/{prompt_name}.bin")["prompt_embeddings"], origin_weigts=torch.load(f"soft_prompts/{origin_prompt}.bin")["prompt_embeddings"]) for prompt_name in dataset_names] for origin_prompt in pa_config.origin_prompts}

def create_task_combinations(task_prompts : List[TaskPrompt], n : int = 2) -> List[TaskPrompt]:
    tp_cominations = itertools.combinations(task_prompts, n)

    return [reduce(operator.add, tp) for tp in tp_cominations]


parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pa_config = parser.parse_toml_file("configs/addition.toml")

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

_, _, test_datasets = preprocessor.get_data()

tp_per_origin = get_task_prompts(pa_config=pa_config, dataset_names=data_args.dataset_names)

print(tp_per_origin)

for tp in create_task_combinations(tp_per_origin["origin"]):
    print(tp.task_name)

for origin_prompt in tp_per_origin:
    origin_weights = torch.load(f"soft_prompts/{origin_prompt}.bin")["prompt_embeddings"]
    model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(origin_weights)

    print(model.prompt_encoder.default.embedding.weight)

    evaluator = ArithmeticsEvaluator(
        task_prompts=tp_per_origin[origin_prompt] + create_task_combinations(tp_per_origin[origin_prompt]),
        model=model,
        datasets=test_datasets,
        training_args=training_args,
        tokenizer=tokenizer,
        origin_weights=origin_weights
    )
    results = evaluator.run()

    df = pd.DataFrame.from_dict(results)
    df = df.groupby(["tasks"], as_index=False).first()
    df.to_csv("./resuts.csv")