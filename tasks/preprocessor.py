from typing import List
from transformers import AutoTokenizer
from datasets import concatenate_datasets

from .tasks import AutoTask
from args import DataTrainingArguments, TrainingArguments

import functools


class Preprocessor:
    def __init__(
        self,
        tasks: List[str],
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
    ):
        self.tasks = tasks
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
        )

    def preprocess_function(self, examples, max_target_length: int):
        padding = "max_length"

        inputs = self.tokenizer(
            examples["source"],
            max_length=self.data_args.max_source_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            examples["target"],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )

        inputs["labels"] = labels["input_ids"]

        return inputs

    def get_data(self):
        cols_to_remove = ["source", "target"]

        train_dataset = None
        valid_datasets = None
        test_datasets = None

        max_target_lengths = [
            AutoTask.get(dataset_name).get_max_target_length(
                self.tokenizer, default_max_length=self.data_args.max_target_length
            )
            for dataset_name in self.tasks
        ]

        print(f"Max target lengths: {max_target_lengths}")

        if self.training_args.do_train:
            train_datasets = [
                AutoTask.get(dataset_name).get(
                    split="train",
                    split_validation_test=self.data_args.split_validation_test,
                    add_prefix=True,
                    n_obs=(self.data_args.max_train_samples),
                )
                for dataset_name in self.tasks
            ]

            for i, train_dataset in enumerate(train_datasets):
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(
                        self.preprocess_function,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on train_dataset",
                )

                train_datasets[i] = train_datasets[i].remove_columns(cols_to_remove)

            train_dataset = concatenate_datasets(train_datasets)

        if self.training_args.do_eval:
            valid_datasets = {
                dataset_name: AutoTask.get(dataset_name).get(
                    split="validation",
                    split_validation_test=self.data_args.split_validation_test,
                    add_prefix=True,
                    n_obs=(self.data_args.max_valid_samples),
                )
                for dataset_name in self.tasks
            }

            for i, name in enumerate(valid_datasets):
                # print(valid_datasets[name][0])
                valid_datasets[name] = valid_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on valid_dataset",
                )

                valid_datasets[name] = valid_datasets[name].remove_columns(
                    cols_to_remove
                )

        if self.training_args.do_test:
            test_datasets = {
                dataset_name: AutoTask.get(dataset_name).get(
                    split="test",
                    split_validation_test=self.data_args.split_validation_test,
                    add_prefix=True,
                    n_obs=(self.data_args.max_test_samples),
                )
                for dataset_name in self.tasks
            }

            for i, name in enumerate(test_datasets):
                # print(test_datasets[name][0])
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on test_dataset",
                )

                test_datasets[name] = test_datasets[name].remove_columns(cols_to_remove)

        return train_dataset, valid_datasets, test_datasets