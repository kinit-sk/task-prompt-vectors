import functools
import datasets
import numpy as np
import torch

from .type import AutoType
from .utils import round_stsb_target

from collections import OrderedDict, defaultdict
from typing import Mapping

from typing import List

from evaluate import Metric

from metrics import (
    f1_score_with_invalid,
    accuracy_with_invalid,
    spearmanr,
    pearsonr,
)


class AbstractTask:
    name: str = NotImplemented
    preprocessor: function = NotImplemented
    formater: function = NotImplemented
    metrics: List[Metric] = NotImplemented
    metric_names: List[str] = NotImplemented
    config = NotImplemented
    dataset_config_name = NotImplemented
    seed = NotImplemented
    labels_list = None
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    small_datasets_without_all_splits = [
        "wnli",
        "stsb",
        "scitail",
    ]
    large_data_without_all_splits = [
        "qqp",
        "qnli",
        "sst2",
        "snli",
        "amazon_polarity",
        "yelp_polarity",
        "winogrande",
    ]

    def __init__(self, seed=42):
        self.dataset_config_name = "en"
        self.seed = seed

    def postprocessor(
        self, preds, labels, tokenizer, ignore_pad_token_for_loss, data_info=None
    ):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        return decoded_preds, decoded_labels

    # get maximum token lenght from labels
    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, add_prefix):
        return dataset.map(
            functools.partial(self.preprocessor, add_prefix=add_prefix),
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc=f"Running {self.name}_preprocessor on dataset",
        )

    def load_dataset(self, split: int):
        return datasets.load_dataset(
            self.name, self.dataset_config_name, split=split, script_version="master"
        )

    def get(
        self,
        split,
        task_type="seq_2_seq_lm",
        add_prefix=True,
        n_obs=None,
        split_validation_test=False,
    ):
        self.formater = AutoType.get(task_type).formater
        if (
            split_validation_test
            and self.name in self.small_datasets_without_all_splits
            and split != "train"
        ):
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(
                split, dataset, validation_size=len(dataset) // 2
            )
            dataset = self.subsample(dataset, n_obs, indices)

        elif (
            split_validation_test
            and self.name in self.large_data_without_all_splits
            and split != "test"
        ):
            dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)

        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)

            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)

        return self.map_dataset(dataset, add_prefix)


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence", example["sentence"]]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "question:",
            example["question"],
            "sentence:",
            example["sentence"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", "wnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [str(example["label"])]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metrics = [accuracy_with_invalid, f1_score_with_invalid]
    metric_names = ["accuracy_with_invalid", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "question1:",
            example["question1"],
            "question2:",
            example["question2"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]

    metrics = [pearsonr, spearmanr]
    metric_names = ["pearsonr", "spearmanr"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]

        label_texts = [str(round_stsb_target(example["label"]))]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class WinoGrande(AbstractTask):
    name = "winogrande"
    labels_list = ["0", "1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]

    def load_dataset(self, split):
        return datasets.load_dataset("winogrande", "winogrande_xl", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence:",
            example["sentence"],
            "option0:",
            example["option1"],
            "option1:",
            example["option1"],
        ]
        label_texts = [str(int(example["answer"]) - 1)]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset("scitail", "snli_format", split=split)

    def preprocessor(self, example, add_prefix=True):
        label2id = {"entailment": "0", "neutral": "1"}
        input_texts = [
            "premise:",
            example["sentence1"],
            "hypothesis:",
            example["sentence2"],
        ]
        label_texts = [label2id[example["gold_label"]]]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset("yelp_polarity")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [str(example["label"])]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


TASK_MAPPING = OrderedDict(
    [
        ("sst2", SST2),
        ("qnli", QNLI),
        ("mnli", MNLI),
        ("wnli", WNLI),
        ("qqp", QQP),
        ("stsb", STSB),
        ("winogrande", WinoGrande),
        ("scitail", SciTail),
        ("yelp_polarity", YelpPolarity),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](seed=seed)

        raise ValueError(
            f"Unrecognized task {task} for AutoTask.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
