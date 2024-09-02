from evaluator import ArithmeticsEvaluator
from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig, TaskPrompt
from tasks import Preprocessor
from utils import get_task_prompt_vectors, get_task_prompts, create_heatmaps

import torch


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import TaskType, PromptTuningConfig, get_peft_model

import os

import numpy as np

from datetime import datetime

from typing import Callable, Dict, List

from torch.nn.functional import cosine_similarity

from tqdm import tqdm

from sklearn.cluster import KMeans

import pandas as pd

from collections import defaultdict

mapping = {
    "dbpedia_text_dbpedia_text": "DBPedia DBPedia",
    "dbpedia_text_mnli_text": "DBPedia MNLI",
    "dbpedia_text_qnli_text": "DBPedia QNLI",
    "dbpedia_text_sst2_text": "DBPedia SST2",
    "dbpedia_text_trec_coarse_text": "DBPedia TREC",
    "dbpedia_text_yelp_polarity_text": "DBPedia Yelp",
    "mnli_text_dbpedia_text": "MNLI DBPedia",
    "mnli_text_mnli_text": "MNLI MNLI",
    "mnli_text_qnli_text": "MNLI QNLI",
    "mnli_text_sst2_text": "MNLI SST2",
    "mnli_text_trec_coarse_text": "MNLI TREC",
    "mnli_text_yelp_polarity_text": "MNLI Yelp",
    "qnli_text_dbpedia_text": "QNLI DBPedia",
    "qnli_text_mnli_text": "QNLI MNLI",
    "qnli_text_qnli_text": "QNLI QNLI",
    "qnli_text_sst2_text": "QNLI SST2",
    "qnli_text_trec_coarse_text": "QNLI TREC",
    "qnli_text_yelp_polarity_text": "QNLI Yelp",
    "sst2_text_dbpedia_text": "SST2 DBPedia",
    "sst2_text_mnli_text": "SST2 MNLI",
    "sst2_text_qnli_text": "SST2 QNLI",
    "sst2_text_sst2_text": "SST2 SST2",
    "sst2_text_trec_coarse_text": "SST2 TREC",
    "sst2_text_yelp_polarity_text": "SST2 Yelp",
    "trec_coarse_text_dbpedia_text": "TREC DBPedia",
    "trec_coarse_text_mnli_text": "TREC MNLI",
    "trec_coarse_text_qnli_text": "TREC QNLI",
    "trec_coarse_text_sst2_text": "TREC SST2",
    "trec_coarse_text_trec_coarse_text": "TREC TREC",
    "trec_coarse_text_yelp_polarity_text": "TREC Yelp",
    "yelp_polarity_text_dbpedia_text": "Yelp DBPedia",
    "yelp_polarity_text_mnli_text": "Yelp MNLI",
    "yelp_polarity_text_qnli_text": "Yelp QNLI",
    "yelp_polarity_text_sst2_text": "Yelp SST2",
    "yelp_polarity_text_trec_coarse_text": "Yelp TREC",
    "yelp_polarity_text_yelp_polarity_text": "Yelp Yelp",
}


def average_diff(t1, t2):
    return torch.abs(t1 - t2).mean()


def l2_norm(t1, t2):
    return torch.abs(torch.norm(t1) - torch.norm(t2))


def cosine_sim(t1, t2):
    return cosine_similarity(t1.flatten(), t2.flatten(), dim=0)


def remove_duplicates(keys):
    unique_keys = set()
    result = []

    for key in keys:
        key_parts = mapping[key].split(" ")
        reverse_key = " ".join(key_parts[::-1])
        # print(reverse_key, unique_keys)

        if mapping[key] not in unique_keys and reverse_key not in unique_keys:
            unique_keys.add(mapping[key])
            result.append(key)

    return result


def get_tpv_comparison(
    data_args: DataTrainingArguments,
    task_prompt_vectors: Dict[str, List[TaskPrompt]],
    function: Callable,
):
    cross_origin_comparisons = {}
    for i in tqdm(range(len(data_args.dataset_names))):
        # print(data_args.dataset_names[i])
        cross_origin_comparisons[data_args.dataset_names[i]] = []
        for o1 in task_prompt_vectors:
            cross_origin_comparisons[data_args.dataset_names[i]].append([])
            for o2 in task_prompt_vectors:
                # print(o1, o2)
                tpv1 = task_prompt_vectors[o1][i]
                tpv2 = task_prompt_vectors[o2][i]

                # print(tpv1.task_name, tpv2.task_name)
                cross_origin_comparisons[data_args.dataset_names[i]][-1].append(
                    function(tpv1.prompt, tpv2.prompt).item()
                )

        cross_origin_comparisons[data_args.dataset_names[i]] = torch.Tensor(
            cross_origin_comparisons[data_args.dataset_names[i]]
        )
        # cross_origin_comparisons[
        #     data_args.dataset_names[i]
        # ] /= cross_origin_comparisons[data_args.dataset_names[i]].max()

    return cross_origin_comparisons


# get comparison cross task cross origin
def get_tpv_ct_comparison(
    data_args: DataTrainingArguments,
    task_prompt_vectors: Dict[str, List[TaskPrompt]],
    function: Callable,
):
    cross_origin_comparisons = {}
    for i in tqdm(range(len(data_args.dataset_names))):
        for j in tqdm(range(len(data_args.dataset_names))):
            # print(f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}")
            cross_origin_comparisons[
                f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
            ] = []
            for o1 in task_prompt_vectors:
                cross_origin_comparisons[
                    f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
                ].append([])
                for o2 in task_prompt_vectors:
                    # print(o1, o2)
                    tpv1 = task_prompt_vectors[o1][i]
                    tpv2 = task_prompt_vectors[o2][j]

                    # print(tpv1.task_name, tpv2.task_name)
                    cross_origin_comparisons[
                        f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
                    ][-1].append(function(tpv1.prompt, tpv2.prompt).item())

            cross_origin_comparisons[
                f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
            ] = torch.Tensor(
                cross_origin_comparisons[
                    f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
                ]
            )

    return cross_origin_comparisons


def get_task_ct_cs(
    data_args: DataTrainingArguments,
    task_prompts: Dict[str, List[torch.Tensor]],
):
    cross_origin_task_cs = {}
    for i in tqdm(range(len(data_args.dataset_names))):
        for j in tqdm(range(len(data_args.dataset_names))):
            # print(f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}")
            cross_origin_task_cs[
                f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
            ] = []
            for o1 in task_prompts:
                cross_origin_task_cs[
                    f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
                ].append([])
                for o2 in task_prompts:
                    # print(o1, o2)
                    tp1 = task_prompts[o1][i]
                    tp2 = task_prompts[o2][j]

                    cross_origin_task_cs[
                        f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
                    ][-1].append(cosine_sim(tp1, tp2).item())

            cross_origin_task_cs[
                f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
            ] = torch.Tensor(
                cross_origin_task_cs[
                    f"{data_args.dataset_names[i]}_{data_args.dataset_names[j]}"
                ]
            )

    return cross_origin_task_cs


def get_task_cs(
    data_args: DataTrainingArguments,
    task_prompts: Dict[str, List[torch.Tensor]],
):
    cross_origin_task_cs = {}
    for i in tqdm(range(len(data_args.dataset_names))):
        # print(data_args.dataset_names[i])
        cross_origin_task_cs[data_args.dataset_names[i]] = []
        for o1 in task_prompts:
            cross_origin_task_cs[data_args.dataset_names[i]].append([])
            for o2 in task_prompts:
                # print(o1, o2)
                tp1 = task_prompts[o1][i]
                tp2 = task_prompts[o2][i]

                cross_origin_task_cs[data_args.dataset_names[i]][-1].append(
                    cosine_sim(tp1, tp2).item()
                )

        cross_origin_task_cs[data_args.dataset_names[i]] = torch.Tensor(
            cross_origin_task_cs[data_args.dataset_names[i]]
        )

    return cross_origin_task_cs


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pa_config = parser.parse_toml_file(
    "configs/cross_origin.toml"
)
data_args.dataset_names = sorted(data_args.dataset_names)


# create heatmaps for task prompt vectors
task_prompt_vectors = get_task_prompt_vectors(
    pa_config=pa_config, dataset_names=data_args.dataset_names
)

name_func_map = {"diff": average_diff, "l2": l2_norm, "cosine": cosine_sim}

for fname in name_func_map:
    print(fname)
    cross_origin_comparisons = get_tpv_comparison(
        data_args, task_prompt_vectors, name_func_map[fname]
    )

    create_heatmaps(
        cross_origin_comparisons,
        filename_prefix=f"tpv_{fname}_{timestamp}",
        save_dir=f"./visuals/{timestamp}",
    )

dup_tpv_ct_cs = get_tpv_ct_comparison(data_args, task_prompt_vectors, cosine_sim)
# print(tpv_ct_cs)

tpv_ct_cs = dict(
    filter(
        lambda x: x[0] in remove_duplicates(dup_tpv_ct_cs.keys()), dup_tpv_ct_cs.items()
    )
)

print(f"tpv_ct_cs_{timestamp}")
create_heatmaps(
    tpv_ct_cs,
    filename_prefix=f"tpv_ct_cs_{timestamp}",
    save_dir=f"./visuals/{timestamp}",
    n_rows=7,
    figsize=(25, 40),
)

tpv_ct_cs = dup_tpv_ct_cs
avg_ct_co_tpv_mean = defaultdict(lambda: defaultdict(dict))
avg_ct_co_tpv_std = defaultdict(lambda: defaultdict(dict))
print("average cross origin, cross tasks, task prompt vector cosine similarity:")
for dataset_name in tpv_ct_cs:
    n = len(tpv_ct_cs[dataset_name])

    if mapping[dataset_name].split(" ")[0] == mapping[dataset_name].split(" ")[1]:
        no_diag = tpv_ct_cs[dataset_name].masked_select(
            torch.tril(torch.ones(n, n, dtype=bool), diagonal=-1)
        )
    else:
        no_diag = tpv_ct_cs[dataset_name]

    print(no_diag.shape)
    # print(dataset_name, np.round(no_diag.mean().item(), 2), np.round(no_diag.std().item(), 2))
    avg_ct_co_tpv_mean[mapping[dataset_name].split(" ")[0]][
        mapping[dataset_name].split(" ")[1]
    ] = np.round(no_diag.mean().item(), 2)
    avg_ct_co_tpv_std[mapping[dataset_name].split(" ")[0]][
        mapping[dataset_name].split(" ")[1]
    ] = np.round(no_diag.std().item(), 2)


df_mean = pd.DataFrame.from_dict(avg_ct_co_tpv_mean)
df_std = pd.DataFrame.from_dict(avg_ct_co_tpv_std)
print(df_mean)

df_mean.to_csv("avg_ct_co_tpv_mean.csv")
df_std.to_csv("avg_ct_co_tpv_std.csv")

task_prompts = get_task_prompts(
    pa_config=pa_config, dataset_names=data_args.dataset_names
)

dup_task_ct_cs = get_task_ct_cs(data_args, task_prompts)
task_ct_cs = dict(
    filter(
        lambda x: x[0] in remove_duplicates(dup_task_ct_cs.keys()),
        dup_task_ct_cs.items(),
    )
)

print(f"task_ct_cs_{timestamp}")
create_heatmaps(
    task_ct_cs,
    filename_prefix=f"task_ct_cs_{timestamp}",
    save_dir=f"./visuals/{timestamp}",
    n_rows=7,
    figsize=(25, 40),
)

task_ct_cs = dup_task_ct_cs
avg_ct_co_task_mean = defaultdict(lambda: defaultdict(dict))
avg_ct_co_task_std = defaultdict(lambda: defaultdict(dict))
print("average cross origin, cross tasks, task prompt cosine similarity:")
for dataset_name in task_ct_cs:
    n = len(task_ct_cs[dataset_name])
    if mapping[dataset_name].split(" ")[0] == mapping[dataset_name].split(" ")[1]:
        no_diag = task_ct_cs[dataset_name].masked_select(
            torch.tril(torch.ones(n, n, dtype=bool), diagonal=-1)
        )
    else:
        no_diag = tpv_ct_cs[dataset_name]

    # print(no_diag)
    # print(dataset_name, np.round(no_diag.mean().item(), 2), np.round(no_diag.std().item(), 2))
    avg_ct_co_task_mean[mapping[dataset_name].split(" ")[0]][
        mapping[dataset_name].split(" ")[1]
    ] = np.round(no_diag.mean().item(), 2)
    avg_ct_co_task_std[mapping[dataset_name].split(" ")[0]][
        mapping[dataset_name].split(" ")[1]
    ] = np.round(no_diag.std().item(), 2)


df_mean = pd.DataFrame.from_dict(avg_ct_co_task_mean)
df_std = pd.DataFrame.from_dict(avg_ct_co_task_std)
print(df_mean)

df_mean.to_csv("avg_ct_co_task_mean.csv")
df_std.to_csv("avg_ct_co_task_std.csv")

# create heatmaps for task prompts
task_prompts = get_task_prompts(
    pa_config=pa_config, dataset_names=data_args.dataset_names
)

cross_origin_task_cs = get_task_cs(data_args, task_prompts)
create_heatmaps(
    cross_origin_task_cs,
    filename_prefix=f"tp_cosine_{timestamp}",
    save_dir=f"./visuals/{timestamp}",
)

print("average cross origin task prompt cosine similarity:")
for dataset_name in cross_origin_task_cs:
    n = len(cross_origin_task_cs[dataset_name])
    no_diag = (
        cross_origin_task_cs[dataset_name]
        .masked_select(~torch.eye(n, dtype=bool))
        .view(n, n - 1)
    )
    print(
        dataset_name,
        np.round(no_diag.mean().item(), 2),
        np.round(no_diag.std().item(), 2),
    )

print("average cross origin task prompt vector cosine similarity:")
cross_origin_tpv_cs = get_tpv_comparison(
    data_args, task_prompt_vectors, name_func_map["cosine"]
)
for dataset_name in cross_origin_tpv_cs:
    n = len(cross_origin_tpv_cs[dataset_name])
    no_diag = (
        cross_origin_tpv_cs[dataset_name]
        .masked_select(~torch.eye(n, dtype=bool))
        .view(n, n - 1)
    )
    print(
        dataset_name,
        np.round(no_diag.mean().item(), 2),
        np.round(no_diag.std().item(), 2),
    )

# cross origin evaluation on datasets
# os.environ["WANDB_PROJECT"] = training_args.wandb_project


# tokenizer = AutoTokenizer.from_pretrained(
#     data_args.data_tokenizer_name_or_path, model_max_length=512, use_fast=True
# )

# model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_name_or_path)

# peft_config = PromptTuningConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM,
#     num_virtual_tokens=pa_config.num_virtual_tokens,
# )

# model = get_peft_model(model, peft_config)
# model.base_model.generation_config.max_new_tokens = data_args.max_target_length


# preprocessor = Preprocessor(data_args.dataset_names, data_args, training_args)

# _, valid_datasets, test_datasets = preprocessor.get_data()

# for o1 in task_prompt_vectors:
#     for o2 in task_prompt_vectors:
#         origin_weights = torch.load(f"soft_prompts/{o1}/{o1}.bin")[
#             "prompt_embeddings"
#         ].to(training_args.device)

#         print(model.prompt_encoder.default.embedding.weight)

#         training_args.run_name = f"addition_{timestamp}_{o1}_{o2}"

#         print(f"origin: {o1} task prompt vectors: {o2}")

#         evaluator = ArithmeticsEvaluator(
#             task_prompts=task_prompt_vectors[o2],
#             model=model,
#             test_datasets=test_datasets,
#             eval_datasets=valid_datasets,
#             training_args=training_args,
#             tokenizer=tokenizer,
#             origin_weights=origin_weights,
#         )
#         results = evaluator.run()

#         results.to_csv(f"./results_{o1}_{o2}_{timestamp}.csv")
