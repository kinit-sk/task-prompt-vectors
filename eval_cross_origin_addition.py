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

from datetime import datetime

# import transformers
# transformers.logging.set_verbosity_debug()

timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

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

_, valid_datasets, test_datasets = preprocessor.get_data()

origin_0_weights = torch.load(f"soft_prompts/origin_0/origin_0.bin")[
    "prompt_embeddings"
]
origin_0_qnli_weights = torch.load(f"soft_prompts/origin_0/qnli.bin")[
    "prompt_embeddings"
]
origin_0_mnli_weights = torch.load(f"soft_prompts/origin_0/mnli.bin")[
    "prompt_embeddings"
]

origin_1_weights = torch.load(f"soft_prompts/origin_1/origin_1.bin")[
    "prompt_embeddings"
]
origin_1_qnli_weights = torch.load(f"soft_prompts/origin_1/qnli.bin")[
    "prompt_embeddings"
]
origin_1_mnli_weights = torch.load(f"soft_prompts/origin_1/mnli.bin")[
    "prompt_embeddings"
]


origin_0_qnli = TaskPrompt("qnli", origin_0_qnli_weights, origin_0_weights)
origin_0_mnli = TaskPrompt("mnli", origin_0_mnli_weights, origin_0_weights)

origin_1_qnli = TaskPrompt("qnli", origin_1_qnli_weights, origin_1_weights)
origin_1_mnli = TaskPrompt("mnli", origin_1_mnli_weights, origin_1_weights)

print(model.prompt_encoder.default.embedding.weight)

training_args.run_name = f"cross_addition_origin_0_origin_1_{timestamp}"

evaluator = ArithmeticsEvaluator(
    task_prompts=[origin_0_qnli, origin_1_qnli, origin_0_mnli, origin_1_mnli],
    test_datasets=test_datasets,
    eval_datasets=valid_datasets,
    training_args=training_args,
    tokenizer=tokenizer,
    origin_weights=origin_0_weights,
)
results = evaluator.run()

results.to_csv(f"./results_{timestamp}.csv")
