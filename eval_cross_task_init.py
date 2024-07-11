from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from tasks import Preprocessor
from arithmetics import PromptArithmeticsConfig
from trainer import MultiTaskSeq2SeqTrainer
from tasks import AutoTask

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model

from datetime import datetime

import wandb
import os

import torch
import argparse

timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Eval cross-task PA",
    description="Evaluate cross-task performance of prompt arithmetics.",
)

argparse_parser.add_argument("filename", help="Filename of a config to run.")
args = argparse_parser.parse_args()


parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pt_args = parser.parse_toml_file(args.filename)
print(training_args)

os.environ["WANDB_PROJECT"] = training_args.wandb_project

tokenizer = AutoTokenizer.from_pretrained(
    data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
)

output_dir = training_args.output_dir


for origin_prompt in pt_args.origin_prompts:
    all_init_prompts = [origin_prompt] + pt_args.init_prompts
    if "spot" in pt_args.init_prompts[0]:
        all_init_prompts = pt_args.init_prompts

    for prompt in [origin_prompt] + pt_args.init_prompts:
        training_args.origin_prompt_name = prompt

        if prompt == origin_prompt and "avg" in pt_args.init_prompts[0]:
            continue

        for dataset_name in data_args.dataset_names:
            training_args.train_dataset_names = [dataset_name]

            model = AutoModelForSeq2SeqLM.from_pretrained(
                training_args.model_name_or_path
            )
            model = get_peft_model(model, peft_config=pt_args)

            prompt_w = torch.load(f"soft_prompts/{origin_prompt}/{prompt}.bin")
            if isinstance(prompt_w, dict):
                model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
                    prompt_w["prompt_embeddings"]
                )
            else:
                model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
                    prompt_w
                )

            model.base_model.generation_config.max_new_tokens = (
                data_args.max_target_length
            )

            print("current PT weights:", model.prompt_encoder.default.embedding.weight)

            model.print_trainable_parameters()

            print(f"task: {dataset_name}")

            preprocessor = Preprocessor([dataset_name], data_args, training_args)

            training_args.output_dir = (
                f"{output_dir}_{timestamp}_{dataset_name}_{origin_prompt}_{prompt}"
            )

            if "fewshot" in args.filename:
                training_args.run_name = f"cross_task_fewshot_{data_args.max_train_samples}_{timestamp}_{dataset_name}_{origin_prompt}_{prompt}"
            else:
                training_args.run_name = (
                    f"cross_task_{timestamp}_{dataset_name}_{origin_prompt}_{prompt}"
                )

            train_dataset, valid_datasets, test_datasets = preprocessor.get_data()

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, return_tensors="pt"
            )

            compute_metrics = AutoTask.get(dataset_name).get_compute_metrics(tokenizer)

            trainer = MultiTaskSeq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_datasets,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            if training_args.do_train:
                trainer.train()

            print(
                trainer.evaluate(
                    eval_dataset=test_datasets[dataset_name], metric_key_prefix="test"
                )
            )

            save_name = (
                f"./saves/cross_task_{timestamp}_{dataset_name}_{origin_prompt}_best"
            )
            model.save_pretrained(save_name)

            if wandb.run is not None:
                artifact = wandb.Artifact(name=training_args.run_name, type="weights")
                artifact.add_dir(local_path=save_name)
                wandb.run.log_artifact(artifact)
                wandb.log(data={})

                wandb.finish()
