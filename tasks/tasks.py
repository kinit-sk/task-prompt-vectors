import functools
import datasets
import numpy as np
import torch

from .type import AutoType

from collections import OrderedDict
from typing import Mapping

from typing import List, Dict, Callable

from evaluate import Metric
from datasets import Dataset

from metrics import (
    exact_match,
    macro_f1,
    f1,
    squad_v2_metric,
    pearsonr,
    spearmanr,
    matthews_correlation,
    round_stsb_target,
)

from transformers import EvalPrediction

import re


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
        "cola",
        "rte",
        "mrpc",
        "trec_fine",
        "trec_coarse",
        "wnli",
        "sst5",
    ]
    large_data_without_all_splits = [
        "math",
        "qnli",
        "sst2",
        "mnli",
        "yelp_polarity",
        "dbpedia",
        "scitail",
        "snli",
        "ag_news",
        "yahoo",
        "imdb",
        "squad_v2",
        "mmlu",
        "hotpot_qa",
        "qqp",
    ]
    id2label = NotImplemented
    label_column_name = NotImplemented

    def __init__(self, seed=42):
        self.dataset_config_name = "en"
        self.seed = seed

    def postprocessor(self, preds, labels, tokenizer):
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, return_tensors="pt"
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, return_tensors="pt"
        )

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

    # generates indices of the dataset randomly with seed (if same seed and data provided we will still get the same shuffle, no matter how many times initialized)
    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset: Dataset, n_obs=None):
        num_samples = len(dataset)

        if n_obs >= num_samples:
            return dataset

        return dataset.train_test_split(
            train_size=n_obs / num_samples,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
        )["train"]

    def get_splits(self, split, dataset: Dataset, validation_size):
        if split == "validation":
            return dataset.train_test_split(
                train_size=validation_size,
                test_size=1 - validation_size,
                seed=self.seed,
                stratify_by_column=self.label_column_name,
                shuffle=True,
            )["train"]

        return dataset.train_test_split(
            train_size=validation_size,
            test_size=1 - validation_size,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
            shuffle=True,
        )["test"]

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

    def get_compute_metrics(
        self,
        tokenizer,
        postprocess=True,
    ) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics(eval_preds: EvalPrediction) -> Dict:
            preds, labels = eval_preds

            if postprocess:
                preds[preds == -100] = tokenizer.pad_token_id
                labels[labels == -100] = tokenizer.pad_token_id

                decoded_preds, decoded_labels = self.postprocessor(
                    preds, labels, tokenizer
                )
            else:
                decoded_preds = preds
                decoded_labels = labels

            print("compute_metrics:", decoded_preds, decoded_labels)

            metrics = {}
            # TODO: to get rid of the zip, make classes from metrics and add metric name to it
            for m, n in zip(self.metrics, self.metric_names):
                if "f1" in n:
                    metrics.update(m(decoded_preds, decoded_labels, self.labels_list))
                else:
                    metrics.update(m(decoded_preds, decoded_labels))

            return metrics

        return compute_metrics

    def get(
        self,
        split,
        task_type="SEQ_2_SEQ_LM",
        add_prefix=True,
        n_obs=None,
        split_validation_test=False,
    ) -> Dataset:
        self.formater = AutoType.get(task_type).formater
        if (
            split_validation_test
            and self.name.replace("_text", "").replace("_instruct", "")
            in self.small_datasets_without_all_splits
            and split != "train"
        ):
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            dataset = self.get_splits(split, dataset, 0.5)

            if n_obs:
                dataset = self.subsample(dataset, n_obs)

        elif (
            split_validation_test
            and self.name.replace("_text", "").replace("_instruct", "")
            in self.large_data_without_all_splits
            and split != "test"
        ):
            dataset = self.load_dataset(split="train")
            dataset = self.get_splits(split, dataset, 1000 / len(dataset))

            if n_obs:
                dataset = self.subsample(dataset, n_obs)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split).shuffle(seed=self.seed)

            if n_obs:
                dataset = self.subsample(dataset, n_obs)

        return self.map_dataset(dataset, add_prefix)


# Sentiment classification
class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "sst2", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["sentence"]]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class SST2Text(AbstractTask):
    name = "sst2_text"
    labels_list = ["negative", "positive"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "negative", 1: "positive"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "sst2", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["sentence"]]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class SST2TextInstruct(AbstractTask):
    name = "sst2_text_instruct"
    labels_list = ["negative", "positive"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "negative", 1: "positive"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "sst2", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence: {example['sentence']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class SST5(AbstractTask):
    name = "sst5"
    labels_list = ["0", "1", "2", "3", "4"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("SetFit/sst5", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["sentence"]]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class SST5Text(AbstractTask):
    name = "sst5_text"
    labels_list = ["very negative", "negative", "neutral", "positive", "very positive"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    id2label = {
        0: "very negative",
        1: "negative",
        2: "neutral",
        3: "positive",
        4: "very positive",
    }

    label_column_name = "label_text"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("SetFit/sst5", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence", example["sentence"]]
        label_texts = [example[self.label_column_name]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class STSBText(AbstractTask):
    name = "stsb_text"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    metrics = [pearsonr, spearmanr]
    metric_names = ["pearsonr", "spearmanr"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {
        np.round(label, decimals=1): str(np.round(label, decimals=1))
        for label in np.arange(0, 5.2, 0.2)
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "stsb", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [str(round_stsb_target(example["label"]))]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class STSBTextInstruct(AbstractTask):
    name = "stsb_text_instruct"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    metrics = [pearsonr, spearmanr]
    metric_names = ["pearsonr", "spearmanr"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {
        np.round(label, decimals=1): str(np.round(label, decimals=1))
        for label in np.arange(0, 5.2, 0.2)
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "stsb", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence1 and sentence2 pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence1: {example['sentence1']}",
            f"sentence2: {example['sentence2']}",
        ]
        label_texts = [str(round_stsb_target(example["label"]))]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("yelp_polarity")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [str(example[self.label_column_name])]
        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class YelpPolarityText(AbstractTask):
    name = "yelp_polarity_text"
    labels_list = ["negative", "positive"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {0: "negative", 1: "positive"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("yelp_polarity")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [self.id2label[example[self.label_column_name]]]
        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class YelpPolarityTextInstruct(AbstractTask):
    name = "yelp_polarity_text_instruct"
    labels_list = ["negative", "positive"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {0: "negative", 1: "positive"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("yelp_polarity")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence: {example['text']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]
        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class IMDB(AbstractTask):
    name = "imdb"
    labels_list = ["0", "1"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("imdb")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [str(example[self.label_column_name])]
        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class IMDBText(AbstractTask):
    name = "imdb_text"
    labels_list = ["negative", "positive"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {0: "negative", 1: "positive"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("imdb")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [self.id2label[example[self.label_column_name]]]
        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


# Natural language inference
class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "qnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "question:",
            example["question"],
            "sentence:",
            example["sentence"],
        ]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class QNLIText(AbstractTask):
    name = "qnli_text"
    labels_list = ["entailment", "not_entailment"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "not_entailment"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "qnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "question:",
            example["question"],
            "sentence:",
            example["sentence"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class QNLITextInstruct(AbstractTask):
    name = "qnli_text_instruct"
    labels_list = ["entailment", "not entailment"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "not entailment"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "qnli", split=split)

    def preprocessor(self, example, add_prefix=False):
        input_texts = [
            f"Classify the question and sentence pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"question: {example['question']}",
            f"sentence: {example['sentence']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class QQPText(AbstractTask):
    name = "qqp_text"
    labels_list = ["not_duplicate", "duplicate"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "not_duplicate", 1: "duplicate"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "qqp", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "question1:",
            example["question1"],
            "question2:",
            example["question2"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class QQPTextInstruct(AbstractTask):
    name = "qqp_text_instruct"
    labels_list = ["not duplicate", "duplicate"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "not duplicate", 1: "duplicate"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "qqp", split=split)

    def preprocessor(self, example, add_prefix=False):
        input_texts = [
            f"Classify the question1 and question2 pair into following labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"question1: {example['question1']}",
            f"question2: {example['question2']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "mnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class MNLIText(AbstractTask):
    name = "mnli_text"
    labels_list = ["entailment", "neutral", "contradiction"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "mnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class MNLITextInstruct(AbstractTask):
    name = "mnli_text_instruct"
    labels_list = ["entailment", "neutral", "contradiction"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "mnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the premise and hypothesis pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"premise: {example['premise']}",
            f"hypothesis: {example['hypothesis']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class MRPCText(AbstractTask):
    name = "mrpc_text"
    labels_list = ["not_equivalent", "equivalent"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "not_equivalent", 1: "equivalent"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "mrpc", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class MRPCTextInstruct(AbstractTask):
    name = "mrpc_text_instruct"
    labels_list = ["not equivalent", "equivalent"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "not equivalent", 1: "equivalent"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "mrpc", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence1 and sentence2 pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence1: {example['sentence1']}",
            f"sentence2: {example['sentence2']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class RTEText(AbstractTask):
    name = "rte_text"
    labels_list = ["entailment", "not_entailment"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "not_entailment"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "rte", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class RTETextInstruct(AbstractTask):
    name = "rte_text_instruct"
    labels_list = ["entailment", "not entailment"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "not entailment"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "rte", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence1 and sentence2 pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence1: {example['sentence1']}",
            f"sentence2: {example['sentence2']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class COLAText(AbstractTask):
    name = "cola_text"
    labels_list = ["unacceptable", "acceptable"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "unacceptable", 1: "acceptable"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "cola", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence:",
            example["sentence"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class COLATextInstruct(AbstractTask):
    name = "cola_text_instruct"
    labels_list = ["unacceptable", "acceptable"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "unacceptable", 1: "acceptable"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "cola", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence: {example['sentence']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    label_column_name = "gold_label"

    def load_dataset(self, split):
        return datasets.load_dataset(
            "scitail", "snli_format", split=split
        ).class_encode_column(self.label_column_name)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["sentence1"],
            "hypothesis:",
            example["sentence2"],
        ]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class SciTailText(AbstractTask):
    name = "scitail_text"
    labels_list = ["entailment", "neutral"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    label_column_name = "gold_label"
    id2label = {0: "entailment", 1: "neutral"}

    def load_dataset(self, split):
        return datasets.load_dataset(
            "scitail", "snli_format", split=split
        ).class_encode_column(self.label_column_name)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["sentence1"],
            "hypothesis:",
            example["sentence2"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    label_column_name = "label"

    def load_dataset(self, split):
        return datasets.load_dataset("snli", split=split).filter(
            lambda x: x["label"] != -1
        )

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class SNLIText(AbstractTask):
    name = "snli_text"
    labels_list = ["entailment", "neutral", "contradiction"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    label_column_name = "label"
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def load_dataset(self, split):
        return datasets.load_dataset("snli", split=split).filter(
            lambda x: x["label"] != -1
        )

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"

    def load_dataset(self, split):
        return datasets.load_dataset("glue", "wnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [str(example[self.label_column_name])]
        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class WNLIText(AbstractTask):
    name = "wnli_text"
    labels_list = ["not_entailment", "entailment"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "not_entailment", 1: "entailment"}

    def load_dataset(self, split):
        return datasets.load_dataset("glue", "wnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]
        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


# Multi task question classification
class TRECFine(AbstractTask):
    name = "trec_fine"
    labels_list = [str(i) for i in list(range(50))]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "test"}
    label_column_name = "fine_label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class TRECFineText(AbstractTask):
    name = "trec_fine_text"
    labels_list = [
        "abbreviation",
        "expression abbreviated",
        "an animal",
        "an organ of the body",
        "a color",
        "creative piece",
        "currency",
        "disease or medicine",
        "event",
        "food",
        "musical instrument",
        "language",
        "letter",
        "other entity",
        "plant",
        "product",
        "religion",
        "sport",
        "substance",
        "symbol",
        "technique",
        "term",
        "vehicle",
        "word",
        "definition",
        "description",
        "manner of action",
        "reason",
        "group",
        "individual",
        "title",
        "description",
        "city",
        "country",
        "mountain",
        "other location",
        "state",
        "code",
        "count",
        "date",
        "distance",
        "price",
        "order",
        "other number",
        "period of time",
        "percentage",
        "speed",
        "temperature",
        "size",
        "weight",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "test"}
    label_column_name = "fine_label"
    id2label = {
        0: "abbreviation",
        1: "expression abbreviated",
        2: "an animal",
        3: "an organ of the body",
        4: "a color",
        5: "creative piece",
        6: "currency",
        7: "disease or medicine",
        8: "event",
        9: "food",
        10: "musical instrument",
        11: "language",
        12: "letter",
        13: "other entity",
        14: "plant",
        15: "product",
        16: "religion",
        17: "sport",
        18: "substance",
        19: "symbol",
        20: "technique",
        21: "term",
        22: "vehicle",
        23: "word",
        24: "definition",
        25: "description",
        26: "manner of action",
        27: "reason",
        28: "group",
        29: "individual",
        30: "title",
        31: "description",
        32: "city",
        33: "country",
        34: "mountain",
        35: "other location",
        36: "state",
        37: "code",
        38: "count",
        39: "date",
        40: "distance",
        41: "price",
        42: "order",
        43: "other number",
        44: "period of time",
        45: "percentage",
        46: "speed",
        47: "temperature",
        48: "size",
        49: "weight",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["text"]]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class TRECCoarse(AbstractTask):
    name = "trec_coarse"
    labels_list = [str(i) for i in list(range(6))]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "test"}
    label_column_name = "coarse_label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["What is this question asking for:", example["text"]]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class TRECCoarseText(AbstractTask):
    name = "trec_coarse_text"
    labels_list = [
        "Abbreviation",
        "Entity",
        "Description",
        "Person",
        "Location",
        "Quantity",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "test"}
    label_column_name = "coarse_label"
    id2label = {
        0: "Abbreviation",
        1: "Entity",
        2: "Description",
        3: "Person",
        4: "Location",
        5: "Quantity",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["What is this question asking for:", example["text"]]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class TRECCoarseTextInstruct(AbstractTask):
    name = "trec_coarse_text_instruct"
    labels_list = [
        "Abbreviation",
        "Entity",
        "Description",
        "Person",
        "Location",
        "Quantity",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "test"}
    label_column_name = "coarse_label"
    id2label = {
        0: "Abbreviation",
        1: "Entity",
        2: "Description",
        3: "Person",
        4: "Location",
        5: "Quantity",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the question into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"question: {example['text']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class DBPEDIA(AbstractTask):
    name = "dbpedia"
    labels_list = [str(i) for i in list(range(14))]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("dbpedia_14", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "What is the category for following text: title:",
            example["title"],
            "content:",
            example["content"],
        ]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class DBPEDIAText(AbstractTask):
    name = "dbpedia_text"
    labels_list = [
        "Company",
        "Educational Institution",
        "Artist",
        "Athlete",
        "Office Holder",
        "Mean Of Transportation",
        "Building",
        "Natural Place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written Work",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {
        0: "Company",
        1: "Educational Institution",
        2: "Artist",
        3: "Athlete",
        4: "Office Holder",
        5: "Mean Of Transportation",
        6: "Building",
        7: "Natural Place",
        8: "Village",
        9: "Animal",
        10: "Plant",
        11: "Album",
        12: "Film",
        13: "Written Work",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("dbpedia_14", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "What is the category for following text: title:",
            example["title"],
            "content:",
            example["content"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class DBPEDIATextInstruct(AbstractTask):
    name = "dbpedia_text_instruct"
    labels_list = [
        "Company",
        "Educational Institution",
        "Artist",
        "Athlete",
        "Office Holder",
        "Mean Of Transportation",
        "Building",
        "Natural Place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written Work",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {
        0: "Company",
        1: "Educational Institution",
        2: "Artist",
        3: "Athlete",
        4: "Office Holder",
        5: "Mean Of Transportation",
        6: "Building",
        7: "Natural Place",
        8: "Village",
        9: "Animal",
        10: "Plant",
        11: "Album",
        12: "Film",
        13: "Written Work",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("dbpedia_14", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the title and content pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"title: {example['title']}",
            f"content: {example['content']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class AGNews(AbstractTask):
    name = "ag_news"
    labels_list = [str(i) for i in list(range(4))]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("ag_news", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "What is the category for following text:",
            example["text"],
        ]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class AGNewsText(AbstractTask):
    name = "ag_news_text"
    labels_list = [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("ag_news", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "What is the category for following text:",
            example["text"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class Yahoo(AbstractTask):
    name = "yahoo"
    labels_list = [str(i) for i in list(range(10))]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "topic"

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("yahoo_answers_topics", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "What is the category for following text: title:",
            example["question_title"],
            "content:",
            example["question_content"],
            "answer:",
            example["best_answer"],
        ]
        label_texts = [str(example[self.label_column_name])]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class YahooText(AbstractTask):
    name = "yahoo_text"
    labels_list = [
        "Society",
        "Science",
        "Health",
        "Education",
        "Computers",
        "Sports",
        "Business",
        "Entertainment",
        "Family",
        "Politics",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "topic"
    id2label = {
        0: "Society",
        1: "Science",
        2: "Health",
        3: "Education",
        4: "Computers",
        5: "Sports",
        6: "Business",
        7: "Entertainment",
        8: "Family",
        9: "Politics",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("yahoo_answers_topics", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "What is the category for following text: title:",
            example["question_title"],
            "content:",
            example["question_content"],
            "answer:",
            example["best_answer"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_text", ""), input_texts, label_texts, add_prefix
        )


class MATHL5InstructEvalAIMO(AbstractTask):
    name = "math_l5_instruct_eval_aimo"
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "train",
        "test": "train",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("AI-MO/aimo-validation-math-level-5", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Solve the Level 5 math problem.",
            "Please answer this question by first reasoning and then providing your answer.\nPresent your reasoning and solution in the following format. Please only write your final answer in the `answer` field, e.g., answer: 42. reasoning: ___, \\n answer: ___",
            f"Problem: {example['problem']}",
        ]
        label_texts = [example["answer"]]

        return self.formater(
            self.name.replace("_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
            generation=True,
        )


class MATHL4InstructEvalAIMO(AbstractTask):
    name = "math_l4_instruct_eval_aimo"
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "train",
        "test": "train",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("AI-MO/aimo-validation-math-level-4", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Solve the Level 4 math problem.",
            "Please answer this question by first reasoning and then providing your answer.\nPresent your reasoning and solution in the following format. Please only write your final answer in the `answer` field, e.g., answer: 42. reasoning: ___, \\n answer: ___",
            f"Problem: {example['problem']}",
        ]
        label_texts = [example["answer"]]

        return self.formater(
            self.name.replace("_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
            generation=True,
        )


class MATHInstruct(AbstractTask):
    name = "math_instruct"
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "test",
        "test": "test",
    }
    label_column_name = None

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset(
            "DigitalLearningGmbH/MATH-lighteval", "default", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Solve the {example['level']} math problem from subject {example['type']}. Provide reasoning followed by the answer to the question.",
            f"Problem: {example['problem']}",
        ]
        label_texts = [example["solution"]]

        return self.formater(
            self.name.replace("_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
            generation=True,
        )


class MMLUInstruct(AbstractTask):
    name = "mmlu_instruct"
    labels_list = ["A", "B", "C", "D"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "auxiliary_train",
        "validation": "validation",
        "test": "test",
    }
    label_column_name = "label"
    id2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "mnli", split=split)

    @staticmethod
    def combine_lists_to_string(keys, values):
        if len(keys) != len(values):
            raise "Error: Lists must be of the same length."

        return ", ".join(f"{key}: {value}" for key, value in zip(keys, values))

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Answer the question by selecting from choices {', '.join(self.labels_list)}. Reply only with the corresponding letter choice as a label.",
            f"question: {example['question']}",
            f"choices: {self.combine_lists_to_string(self.labels_list, example['choices'])}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name.replace("_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class SQuADV2Instruct(AbstractTask):
    name = "squad_v2_instruct"
    metrics = [squad_v2_metric]
    metric_names = ["squad_v2_metric"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = None

    def load_dataset(self, split):
        return datasets.load_dataset("rajpurkar/squad_v2", split=split)

    def preprocessor(self, example, add_prefix):
        input_texts = [
            "Answer the question based on the context. Also provide where the answer starts in the question. Reply in following format: {'text': [<multiple answers>], 'answer_start': [<multiple answer starts>]}.",
            f"question: {example['question']}",
            f"context: {example['context']}",
        ]
        label_texts = [str(example["answers"])]

        return self.formater(
            self.name.replace("_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
            generation=True,
            id=example["id"],
        )


class HotpotQAInstruct(AbstractTask):
    name = "hotpot_qa_instruct"
    metrics = [squad_v2_metric]
    metric_names = ["squad_v2_metric"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = None

    def load_dataset(self, split):
        return datasets.load_dataset(
            "hotpotqa/hotpot_qa", "fullwiki", split=split, trust_remote_code=True
        )

    def preprocessor(self, example, add_prefix):
        input_texts = [
            "Answer the question based on the context. Reply only the answer.",
            f"question: {example['question']}",
            f"context: ",
            "\n\n".join(
                f"{title}\n" + "\n".join(sentences)
                for title, sentences in zip(
                    example["context"]["title"], example["context"]["sentences"]
                )
            ),
        ]
        label_texts = [str(example["answer"])]

        return self.formater(
            self.name.replace("_instruct", ""),
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
            generation=True,
        )


TASK_MAPPING = OrderedDict(
    [
        ("sst2", SST2),
        ("sst2_text", SST2Text),
        ("sst2_text_instruct", SST2TextInstruct),
        ("stsb_text", STSBText),
        ("stsb_text_instruct", STSBTextInstruct),
        ("qnli", QNLI),
        ("qnli_text", QNLIText),
        ("qnli_text_instruct", QNLITextInstruct),
        ("qqp_text", QQPText),
        ("qqp_text_instruct", QQPTextInstruct),
        ("mnli", MNLI),
        ("mnli_text", MNLIText),
        ("mnli_text_instruct", MNLITextInstruct),
        ("mrpc_text", MRPCText),
        ("mrpc_text_instruct", MRPCTextInstruct),
        ("rte_text", RTEText),
        ("rte_text_instruct", RTETextInstruct),
        ("cola_text", COLAText),
        ("cola_text_instruct", COLATextInstruct),
        ("scitail", SciTail),
        ("scitail_text", SciTailText),
        ("snli", SNLI),
        ("snli_text", SNLIText),
        ("wnli", WNLI),
        ("yelp_polarity", YelpPolarity),
        ("yelp_polarity_text", YelpPolarityText),
        ("yelp_polarity_text_instruct", YelpPolarityTextInstruct),
        ("trec_fine", TRECFine),
        ("trec_fine_text", TRECFineText),
        ("trec_coarse", TRECCoarse),
        ("trec_coarse_text", TRECCoarseText),
        ("trec_coarse_text_instruct", TRECCoarseTextInstruct),
        ("dbpedia", DBPEDIA),
        ("dbpedia_text", DBPEDIAText),
        ("dbpedia_text_instruct", DBPEDIATextInstruct),
        ("ag_news", AGNews),
        ("ag_news_text", AGNewsText),
        ("yahoo", Yahoo),
        ("yahoo_text", YahooText),
        ("imdb", IMDB),
        ("imdb_text", IMDBText),
        ("sst5", SST5),
        ("sst5_text", SST2Text),
        ("math_instruct", MATHInstruct),
        ("math_l4_instruct_eval_aimo", MATHL4InstructEvalAIMO),
        ("math_l5_instruct_eval_aimo", MATHL5InstructEvalAIMO),
        ("mmlu_instruct", MMLUInstruct),
        ("squad_v2_instruct", SQuADV2Instruct),
        ("hotpot_qa_instruct", HotpotQAInstruct),
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
