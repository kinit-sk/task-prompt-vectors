from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from tasks import Preprocessor
from arithmetics import PromptArithmeticsConfig
from trainer import MultiTaskSeq2SeqTrainer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model

from datetime import datetime

import wandb
import os

import torch

from tasks import AutoTask

import argparse


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")


argparse_parser = argparse.ArgumentParser(
    prog="Run prompt tuning",
    description="Run prompt tuning to train soft-prompts.",
)

argparse_parser.add_argument("filename", help="Filename of a config to run.")
args = argparse_parser.parse_args()


parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pt_args = parser.parse_toml_file(args.filename)
# print(training_args)

os.environ["WANDB_PROJECT"] = training_args.wandb_project

if pt_args.task_type == "CAUSAL_LM":
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.data_tokenizer_name_or_path,
        model_max_length=512,
        use_fast=True,
        padding_side="left",
    )

    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
else:
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
    )

output_dir = training_args.output_dir

for origin_prompt in pt_args.origin_prompts:
    training_args.origin_prompt_name = origin_prompt

    for dataset_name in data_args.dataset_names:
        training_args.train_dataset_names = dataset_name

        if not isinstance(dataset_name, list):
            training_args.train_dataset_names = [dataset_name]

        if pt_args.task_type == "CAUSAL_LM":
            model = AutoModelForCausalLM.from_pretrained(
                training_args.model_name_or_path, torch_dtype=torch.bfloat16
            ).to("cuda")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                training_args.model_name_or_path
            )
            model.generation_config.max_new_tokens = data_args.max_target_length

        model = get_peft_model(model, peft_config=pt_args)

        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            torch.load(f"saves/{origin_prompt}/{origin_prompt}.bin")[
                "prompt_embeddings"
            ].to("cuda")
        )

        print("current PT weights:", model.prompt_encoder.default.embedding.weight)

        model.print_trainable_parameters()

        print(f"task: {training_args.train_dataset_names}")

        preprocessor = Preprocessor(
            training_args.train_dataset_names, data_args, training_args, pt_args, tokenizer
        )

        training_args.output_dir = f"{output_dir}_{timestamp}_{'_'.join(training_args.train_dataset_names)}_{origin_prompt}"
        training_args.run_name = f"prompt_tuning_{timestamp}_{'_'.join(training_args.train_dataset_names)}_{origin_prompt}"

        train_dataset, valid_datasets, test_datasets = preprocessor.get_data()
        # print(train_dataset, valid_datasets, test_datasets)

        if pt_args.task_type == "CAUSAL_LM":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False, return_tensors="pt"
            )
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, return_tensors="pt"
            )

        compute_metrics = None
        if isinstance(dataset_name, list):
            compute_metrics = {}
            for dm in dataset_name:
                compute_metrics[dm] = AutoTask.get(dm).get_compute_metrics(tokenizer)
        else:
            compute_metrics = AutoTask.get(dataset_name).get_compute_metrics(tokenizer)

        if pt_args.task_type == "CAUSAL_LM":
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=list(valid_datasets.values())[0],
                data_collator=data_collator,
            )
        else:
            trainer = MultiTaskSeq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_datasets,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        trainer.train()

        if isinstance(dataset_name, list):
            for dm in dataset_name:
                print(
                    trainer.evaluate(
                        eval_dataset=test_datasets[dm], metric_key_prefix=f"test_{dm}"
                    )
                )
        else:
            print(
                trainer.evaluate(
                    eval_dataset=test_datasets[dataset_name], metric_key_prefix="test"
                )
            )

        if isinstance(dataset_name, list):
            save_name = f"./saves/prompt_tuning_{timestamp}_{'_'.join(dataset_name)}_{origin_prompt}_best"
        else:
            save_name = (
                f"./saves/prompt_tuning_{timestamp}_{dataset_name}_{origin_prompt}_best"
            )

        model.save_pretrained(save_name)

        if wandb.run is not None:
            artifact = wandb.Artifact(name=training_args.run_name, type="weights")
            artifact.add_dir(local_path=save_name)
            wandb.run.log_artifact(artifact)
            wandb.log(data={})

            wandb.finish()
