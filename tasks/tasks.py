import functools
import datasets
import numpy as np
import torch

from .type import AutoType

from collections import OrderedDict, defaultdict
from typing import Mapping

from typing import List

from evaluate import Metric
from datasets import Dataset

from metrics import accuracy_with_invalid


class AbstractTask:
    name: str = NotImplemented
    preprocessor = NotImplemented
    formater = NotImplemented
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
        "trec_fine",
        "trec_coarse",
    ]
    large_data_without_all_splits = [
        "qnli",
        "sst2",
        "mnli",
        "yelp_polarity",
        "dbpedia",
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

    def map_dataset(self, dataset: Dataset, add_prefix: bool) -> Dataset:
        return dataset.map(
            functools.partial(self.preprocessor, add_prefix=add_prefix),
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc=f"Running {self.name}_preprocessor on dataset",
        )

    def load_dataset(self, split: int) -> Dataset:
        return datasets.load_dataset(
            self.name, self.dataset_config_name, split=split, script_version="master"
        )

    def compute_metrics(self, tokenizer, eval_preds):
        preds, labels = eval_preds

        decoded_preds, decoded_lables = self.postprocessor(preds, labels, tokenizer, ignore_pad_token_for_loss=True)

        return {n: m(decoded_preds, decoded_lables) for n, m in zip(self.metric_names, self.metrics)}

    def get(
        self,
        split,
        task_type="seq_2_seq_lm",
        add_prefix=True,
        n_obs=None,
        split_validation_test=False,
    ) -> Dataset:
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

    def load_dataset(self, split) -> Dataset:
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

    def load_dataset(self, split) -> Dataset:
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

    def load_dataset(self, split) -> Dataset:
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


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy_with_invalid"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("yelp_polarity")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [str(example["label"])]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class TRECFine(AbstractTask):
    name = "trec_fine"
    labels_list = [str(i) for i in list(range(50))]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "validation": "test"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [str(example["fine_label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class TRECCoarse(AbstractTask):
    name = "trec_coarse"
    labels_list = [str(i) for i in list(range(6))]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "validation": "test"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [str(example["coarse_label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class DBPEDIA(AbstractTask):
    name = "dbpedia"
    labels_list = [str(i) for i in list(range(14))]
    metrics = [accuracy_with_invalid]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("dbpedia_14", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "title:",
            example["title"],
            "content:",
            example["content"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


TASK_MAPPING = OrderedDict(
    [
        ("sst2", SST2),
        ("qnli", QNLI),
        ("mnli", MNLI),
        ("yelp_polarity", YelpPolarity),
        ("trec_fine", TRECFine),
        ("trec_coarse", TRECCoarse),
        ("dbpedia", DBPEDIA),
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
