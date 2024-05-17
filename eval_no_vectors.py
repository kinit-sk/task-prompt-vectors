from evaluator import ArithmeticsEvaluator
from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from tasks import Preprocessor, AutoTask
from utils import get_task_prompts

import torch

from trainer import MultiTaskSeq2SeqTrainer


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from peft import TaskType, PromptTuningConfig, get_peft_model

import os

import numpy as np

from datetime import datetime

import wandb


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pa_config = parser.parse_toml_file("configs/addition.toml")
data_args.dataset_names = sorted(data_args.dataset_names)

# cross origin evaluation on datasets
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

task_prompts = get_task_prompts(pa_config, data_args.dataset_names)

for origin_prompt in task_prompts:
    training_args.origin_prompt_name = origin_prompt

    for dataset_name in data_args.dataset_names:
        training_args.train_dataset_names = [dataset_name]

        mnli_weights = torch.load(f"soft_prompts/{origin_prompt}/mnli.bin")
        qnli_weights = torch.load(f"soft_prompts/{origin_prompt}/qnli.bin")

        model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            qnli_weights + mnli_weights
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="pt")

        compute_metrics = AutoTask.get(dataset_name).get_compute_metrics(tokenizer)

        trainer = MultiTaskSeq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            eval_dataset=valid_datasets,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print(
            trainer.evaluate(
                eval_dataset=test_datasets[dataset_name], metric_key_prefix="test"
            )
        )

        save_name = (
            f"./saves/prompt_tuning_{timestamp}_{dataset_name}_{origin_prompt}_best"
        )
        model.save_pretrained(save_name)

        if wandb.run is not None:
            wandb.finish()
