from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from tasks import Preprocessor
from arithmetics import PromptArithmeticsConfig
from trainer import MultiTaskSeq2SeqTrainer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)
from peft import get_peft_model

from datetime import datetime

import wandb
import os

import torch

from tasks import AutoTask


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")


parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pt_args = parser.parse_toml_file("configs/prompt_tuning/single-task/prompt_tuning.toml")
print(training_args)

os.environ["WANDB_PROJECT"] = training_args.wandb_project

tokenizer = AutoTokenizer.from_pretrained(
    data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
)

output_dir = training_args.output_dir

for origin_prompt in pt_args.origin_prompts:
    training_args.origin_prompt_name = origin_prompt

    for dataset_name in data_args.dataset_names:
        training_args.train_dataset_names = [dataset_name]

        model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_name_or_path)

        model.active_adapters = [
            "default"
        ]

        model = get_peft_model(model, peft_config=pt_args)

        model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            torch.load(f"saves/{origin_prompt}/{origin_prompt}.bin")[
                "prompt_embeddings"
            ]
        )
        model.base_model.generation_config.max_new_tokens = data_args.max_target_length

        print("current PT weights:", model.prompt_encoder.default.embedding.weight)
        print(model.active_adapters)

        model.print_trainable_parameters()

        print(f"task: {dataset_name}")

        preprocessor = Preprocessor([dataset_name], data_args, training_args)

        training_args.output_dir = (
            f"{output_dir}_{timestamp}_{dataset_name}_{origin_prompt}_{training_args.model_name_or_path.split("/")[-1].lower()}"
        )
        training_args.run_name = (
            f"prompt_tuning_{timestamp}_{dataset_name}_{origin_prompt}_{training_args.model_name_or_path.split("/")[-1].lower()}"
        )

        train_dataset, valid_datasets, test_datasets = preprocessor.get_data()
        # print(train_dataset, valid_datasets, test_datasets)

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="pt")

        compute_metrics = AutoTask.get(dataset_name).get_compute_metrics(tokenizer, pt_args.task_type)


        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=next(iter(valid_datasets.values())),
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        print(
            trainer.evaluate(
                eval_dataset=test_datasets[dataset_name], metric_key_prefix="test"
            )
        )

        save_name = (
            f"./{training_args.output_dir}_best"
        )
        model.save_pretrained(save_name)

        if wandb.run is not None:
            artifact = wandb.Artifact(name=training_args.run_name, type="weights")
            artifact.add_dir(local_path=save_name)
            wandb.run.log_artifact(artifact)
            wandb.log(data={})

            wandb.finish()