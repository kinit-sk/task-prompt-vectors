from typing import List
from transformers import AutoTokenizer
from datasets import concatenate_datasets

from .tasks import AutoTask
from args import DataTrainingArguments, TrainingArguments
from arithmetics import PromptArithmeticsConfig

import functools

import torch
from torch.nn.utils.rnn import pad_sequence
class Preprocessor:
    def __init__(
        self,
        tasks: List[str],
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        pa_args: PromptArithmeticsConfig,
        tokenizer: AutoTokenizer,
    ):
        self.tasks = tasks
        self.data_args = data_args
        self.training_args = training_args
        self.pa_args = pa_args
        self.tokenizer = tokenizer


    def _update_attention_mask(self, row):
        idx = (row == self.tokenizer.bos_token_id).nonzero()[0]

        return torch.cat((torch.zeros(idx, dtype=torch.long), torch.ones(len(row)-idx, dtype=torch.long)))

    def _move_trailing_pads_to_beginning(self, row):
        first_non_pad_index = len(row) -1

        while first_non_pad_index >= 0 and row[first_non_pad_index] == self.tokenizer.pad_token_id:
            first_non_pad_index -= 1

        non_trailing_pads = row[:first_non_pad_index + 1]
        trailing_pads = row[first_non_pad_index + 1:]

        return torch.cat((trailing_pads, non_trailing_pads))

    def preprocess_function(self, examples, max_target_length: int, include_labels: True):
        padding = "max_length"

        if self.data_args.pad_to_max_length:
            max_target_length = self.data_args.max_source_length

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

        if self.pa_args.task_type == "CAUSAL_LM" and include_labels:
            inputs["labels"] = torch.cat((labels["input_ids"] , torch.full((labels["input_ids"].shape[0], 1), self.tokenizer.eos_token_id)), dim=1)

            inputs["labels"][inputs["labels"] == self.tokenizer.bos_token_id] = -100
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100

            inputs["input_ids"] = torch.cat((inputs["input_ids"], inputs["labels"]), dim=1)
            inputs["input_ids"] = pad_sequence([row[row != -100] for row in inputs["input_ids"]], padding_value=self.tokenizer.pad_token_id, batch_first=True)
            
            inputs["input_ids"] = torch.stack([self._move_trailing_pads_to_beginning(row) for row in inputs["input_ids"]])
            inputs["attention_mask"] = torch.stack([self._update_attention_mask(row) for row in inputs["input_ids"]])
            inputs["labels"] = torch.cat((torch.full((labels["input_ids"].shape[0],inputs["input_ids"].shape[1] - inputs["labels"].shape[1]), -100), inputs["labels"]), dim=1)

        elif self.pa_args.task_type == "CAUSAL_LM":
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
                    task_type=self.pa_args.task_type,
                )
                for dataset_name in self.tasks
            ]

            for i, train_dataset in enumerate(train_datasets):
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(
                        self.preprocess_function,
                        max_target_length=max_target_lengths[i],
                        include_labels=True,
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
                    task_type=self.pa_args.task_type,
                )
                for dataset_name in self.tasks
            }

            for i, name in enumerate(valid_datasets):
                # print(valid_datasets[name][0])
                valid_datasets[name] = valid_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        max_target_length=max_target_lengths[i],
                        include_labels=False,
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
                    task_type=self.pa_args.task_type,
                )
                for dataset_name in self.tasks
            }

            for i, name in enumerate(test_datasets):
                # print(test_datasets[name][0])
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        max_target_length=max_target_lengths[i],
                        include_labels=False,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on test_dataset",
                )

                test_datasets[name] = test_datasets[name].remove_columns(cols_to_remove)

        return train_dataset, valid_datasets, test_datasets
