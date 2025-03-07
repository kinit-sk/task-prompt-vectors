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
    "yelp_polarity_text_math": "Yelp MATH",
    "yelp_polarity_text_squad_v2": "Yelp SQuADv2",
    "trec_coarse_text_math": "TREC MATH",
    "trec_coarse_text_squad_v2": "TREC SQuADv2",
    "sst2_text_math": "SST2 MATH",
    "sst2_text_squad_v2": "SST2 SQuADv2",
    "qnli_text_math": "QNLI MATH",
    "qnli_text_squad_v2": "QNLI SQuADv2",
    "mnli_text_math": "MNLI MATH",
    "mnli_text_squad_v2": "MNLI SQuADv2",
    "dbpedia_text_math": "DBPedia MATH",
    "dbpedia_text_squad_v2": "DBPedia SQuADv2",
    "math_dbpedia_text": "MATH DBPedia",
    "math_mnli_text": "MATH MNLI",
    "math_qnli_text": "MATH QNLI",
    "math_sst2_text": "MATH SST2",
    "math_trec_coarse_text": "MATH TREC",
    "math_yelp_polarity_text": "MATH Yelp",
    "squad_v2_dbpedia_text": "SQuADv2 DBPedia",
    "squad_v2_mnli_text": "SQuADv2 MNLI",
    "squad_v2_qnli_text": "SQuADv2 QNLI",
    "squad_v2_sst2_text": "SQuADv2 SST2",
    "squad_v2_trec_coarse_text": "SQuADv2 TREC",
    "squad_v2_yelp_polarity_text": "SQuADv2 Yelp",
    "math_math": "MATH MATH",
    "squad_v2_squad_v2": "SQuADv2 SQuADv2",
    "math_squad_v2": "MATH SQuADv2",
    "squad_v2_math": "SQuADv2 MATH",
    "rte_text_dbpedia_text": "RTE DBPedia",
    "rte_text_mnli_text": "RTE MNLI",
    "rte_text_qnli_text": "RTE QNLI",
    "rte_text_sst2_text": "RTE SST2",
    "rte_text_trec_coarse_text": "RTE TREC",
    "rte_text_yelp_polarity_text": "RTE Yelp",
    "rte_text_math": "RTE MATH",
    "rte_text_squad_v2": "RTE SQuADv2",
    "rte_text_rte_text": "RTE RTE",
    "rte_text_stsb_text": "RTE STS-B",
    "rte_text_mrpc_text": "RTE MRPC",
    "rte_text_cola_text": "RTE CoLA",
    "rte_text_qqp_text": "RTE QQP",
    "stsb_text_dbpedia_text": "STS-B DBPedia",
    "stsb_text_mnli_text": "STS-B MNLI",
    "stsb_text_qnli_text": "STS-B QNLI",
    "stsb_text_sst2_text": "STS-B SST2",
    "stsb_text_trec_coarse_text": "STS-B TREC",
    "stsb_text_yelp_polarity_text": "STS-B Yelp",
    "stsb_text_math": "STS-B MATH",
    "stsb_text_squad_v2": "STS-B SQuADv2",
    "stsb_text_rte_text": "STS-B RTE",
    "stsb_text_stsb_text": "STS-B STS-B",
    "stsb_text_mrpc_text": "STS-B MRPC",
    "stsb_text_cola_text": "STS-B CoLA",
    "stsb_text_qqp_text": "STS-B QQP",
    "mrpc_text_dbpedia_text": "MRPC DBPedia",
    "mrpc_text_mnli_text": "MRPC MNLI",
    "mrpc_text_qnli_text": "MRPC QNLI",
    "mrpc_text_sst2_text": "MRPC SST2",
    "mrpc_text_trec_coarse_text": "MRPC TREC",
    "mrpc_text_yelp_polarity_text": "MRPC Yelp",
    "mrpc_text_math": "MRPC MATH",
    "mrpc_text_squad_v2": "MRPC SQuADv2",
    "mrpc_text_rte_text": "MRPC RTE",
    "mrpc_text_stsb_text": "MRPC STS-B",
    "mrpc_text_mrpc_text": "MRPC MRPC",
    "mrpc_text_cola_text": "MRPC CoLA",
    "mrpc_text_qqp_text": "MRPC QQP",
    "cola_text_dbpedia_text": "CoLA DBPedia",
    "cola_text_mnli_text": "CoLA MNLI",
    "cola_text_qnli_text": "CoLA QNLI",
    "cola_text_sst2_text": "CoLA SST2",
    "cola_text_trec_coarse_text": "CoLA TREC",
    "cola_text_yelp_polarity_text": "CoLA Yelp",
    "cola_text_math": "CoLA MATH",
    "cola_text_squad_v2": "CoLA SQuADv2",
    "cola_text_rte_text": "CoLA RTE",
    "cola_text_stsb_text": "CoLA STS-B",
    "cola_text_mrpc_text": "CoLA MRPC",
    "cola_text_cola_text": "CoLA CoLA",
    "cola_text_qqp_text": "CoLA QQP",
    "qqp_text_dbpedia_text": "QQP DBPedia",
    "qqp_text_mnli_text": "QQP MNLI",
    "qqp_text_qnli_text": "QQP QNLI",
    "qqp_text_sst2_text": "QQP SST2",
    "qqp_text_trec_coarse_text": "QQP TREC",
    "qqp_text_yelp_polarity_text": "QQP Yelp",
    "qqp_text_math": "QQP MATH",
    "qqp_text_squad_v2": "QQP SQuADv2",
    "qqp_text_rte_text": "QQP RTE",
    "qqp_text_stsb_text": "QQP STS-B",
    "qqp_text_mrpc_text": "QQP MRPC",
    "qqp_text_cola_text": "QQP CoLA",
    "qqp_text_qqp_text": "QQP QQP",
    "dbpedia_text_rte_text": "DBPedia RTE",
    "mnli_text_rte_text": "MNLI RTE",
    "qnli_text_rte_text": "QNLI RTE",
    "sst2_text_rte_text": "SST2 RTE",
    "trec_coarse_text_rte_text": "TREC RTE",
    "yelp_polarity_text_rte_text": "Yelp RTE",
    "math_rte_text": "MATH RTE",
    "squad_v2_rte_text": "SQuADv2 RTE",
    "rte_text_rte_text": "RTE RTE",
    "stsb_text_rte_text": "STS-B RTE",
    "mrpc_text_rte_text": "MRPC RTE",
    "cola_text_rte_text": "CoLA RTE",
    "qqp_text_rte_text": "QQP RTE",
    "dbpedia_text_stsb_text": "DBPedia STS-B",
    "mnli_text_stsb_text": "MNLI STS-B",
    "qnli_text_stsb_text": "QNLI STS-B",
    "sst2_text_stsb_text": "SST2 STS-B",
    "trec_coarse_text_stsb_text": "TREC STS-B",
    "yelp_polarity_text_stsb_text": "Yelp STS-B",
    "math_stsb_text": "MATH STS-B",
    "squad_v2_stsb_text": "SQuADv2 STS-B",
    "rte_text_stsb_text": "RTE STS-B",
    "stsb_text_stsb_text": "STS-B STS-B",
    "mrpc_text_stsb_text": "MRPC STS-B",
    "cola_text_stsb_text": "CoLA STS-B",
    "qqp_text_stsb_text": "QQP STS-B",
    "dbpedia_text_cola_text": "DBPedia CoLA",
    "mnli_text_cola_text": "MNLI CoLA",
    "qnli_text_cola_text": "QNLI CoLA",
    "sst2_text_cola_text": "SST2 CoLA",
    "trec_coarse_text_cola_text": "TREC CoLA",
    "yelp_polarity_text_cola_text": "Yelp CoLA",
    "math_cola_text": "MATH CoLA",
    "squad_v2_cola_text": "SQuADv2 CoLA",
    "rte_text_cola_text": "RTE CoLA",
    "stsb_text_cola_text": "STS-B CoLA",
    "mrpc_text_cola_text": "MRPC CoLA",
    "cola_text_cola_text": "CoLA CoLA",
    "qqp_text_cola_text": "QQP CoLA",
    "dbpedia_text_mrpc_text": "DBPedia MRPC",
    "mnli_text_mrpc_text": "MNLI MRPC",
    "qnli_text_mrpc_text": "QNLI MRPC",
    "sst2_text_mrpc_text": "SST2 MRPC",
    "trec_coarse_text_mrpc_text": "TREC MRPC",
    "yelp_polarity_text_mrpc_text": "Yelp MRPC",
    "math_mrpc_text": "MATH MRPC",
    "squad_v2_mrpc_text": "SQuADv2 MRPC",
    "rte_text_mrpc_text": "RTE MRPC",
    "stsb_text_mrpc_text": "STS-B MRPC",
    "mrpc_text_mrpc_text": "MRPC MRPC",
    "cola_text_mrpc_text": "CoLA MRPC",
    "qqp_text_mrpc_text": "QQP MRPC",
    "dbpedia_text_qqp_text": "DBPedia QQP",
    "mnli_text_qqp_text": "MNLI QQP",
    "qnli_text_qqp_text": "QNLI QQP",
    "sst2_text_qqp_text": "SST2 QQP",
    "trec_coarse_text_qqp_text": "TREC QQP",
    "yelp_polarity_text_qqp_text": "Yelp QQP",
    "math_qqp_text": "MATH QQP",
    "squad_v2_qqp_text": "SQuADv2 QQP",
    "rte_text_qqp_text": "RTE QQP",
    "stsb_text_qqp_text": "STS-B QQP",
    "mrpc_text_qqp_text": "MRPC QQP",
    "cola_text_qqp_text": "CoLA QQP",
    "qqp_text_qqp_text": "QQP QQP",
}


def average_diff(t1, t2):
    return torch.abs(t1 - t2).mean()


def l2_norm(t1, t2):
    return torch.abs(torch.norm(t1) - torch.norm(t2))


def cosine_sim(t1, t2):
    return cosine_similarity(t1.mean(axis=0), t2.mean(axis=0), dim=0)


def remove_duplicates(keys):
    unique_keys = set()
    result = []

    for key in keys:
        key_parts = mapping[key.replace("_instruct", "")].split(" ")
        reverse_key = " ".join(key_parts[::-1])
        # print(reverse_key, unique_keys)

        if (
            mapping[key.replace("_instruct", "")] not in unique_keys
            and reverse_key not in unique_keys
        ):
            unique_keys.add(mapping[key.replace("_instruct", "")])
            result.append(key.replace("_instruct", ""))

    return result


def get_tpv_comparison(
    data_args: DataTrainingArguments,
    task_prompt_vectors: Dict[str, List[TaskPrompt]],
    function: Callable,
):
    cross_origin_comparisons = {}
    for i in tqdm(range(len(data_args.dataset_names))):
        # print(data_args.dataset_names[i].replace('_instruct', ''))
        cross_origin_comparisons[
            data_args.dataset_names[i].replace("_instruct", "")
        ] = []
        for o1 in task_prompt_vectors:
            cross_origin_comparisons[
                data_args.dataset_names[i].replace("_instruct", "")
            ].append([])
            for o2 in task_prompt_vectors:
                # print(o1, o2)
                tpv1 = task_prompt_vectors[o1][i]
                tpv2 = task_prompt_vectors[o2][i]

                # print(tpv1.task_name, tpv2.task_name)
                cross_origin_comparisons[
                    data_args.dataset_names[i].replace("_instruct", "")
                ][-1].append(function(tpv1.prompt, tpv2.prompt).item())

        cross_origin_comparisons[
            data_args.dataset_names[i].replace("_instruct", "")
        ] = torch.Tensor(
            cross_origin_comparisons[
                data_args.dataset_names[i].replace("_instruct", "")
            ]
        )
        # cross_origin_comparisons[
        #     data_args.dataset_names[i].replace('_instruct', '')
        # ] /= cross_origin_comparisons[data_args.dataset_names[i].replace('_instruct', '')].max()

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
            # print(f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}")
            cross_origin_comparisons[
                f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
            ] = []
            for o1 in task_prompt_vectors:
                cross_origin_comparisons[
                    f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
                ].append([])
                for o2 in task_prompt_vectors:
                    print(o1, o2)
                    tpv1 = task_prompt_vectors[o1][i]
                    tpv2 = task_prompt_vectors[o2][j]

                    print(tpv1.task_name, tpv2.task_name)
                    cross_origin_comparisons[
                        f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
                    ][-1].append(function(tpv1.prompt, tpv2.prompt).item())

            cross_origin_comparisons[
                f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
            ] = torch.Tensor(
                cross_origin_comparisons[
                    f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
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
            # print(f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}")
            cross_origin_task_cs[
                f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
            ] = []
            for o1 in task_prompts:
                cross_origin_task_cs[
                    f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
                ].append([])
                for o2 in task_prompts:
                    # print(o1, o2)
                    tp1 = task_prompts[o1][i]
                    tp2 = task_prompts[o2][j]

                    cross_origin_task_cs[
                        f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
                    ][-1].append(cosine_sim(tp1, tp2).item())

            cross_origin_task_cs[
                f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
            ] = torch.Tensor(
                cross_origin_task_cs[
                    f"{data_args.dataset_names[i].replace('_instruct', '')}_{data_args.dataset_names[j].replace('_instruct', '')}"
                ]
            )

    return cross_origin_task_cs


def get_task_cs(
    data_args: DataTrainingArguments,
    task_prompts: Dict[str, List[torch.Tensor]],
):
    cross_origin_task_cs = {}
    for i in tqdm(range(len(data_args.dataset_names))):
        # print(data_args.dataset_names[i].replace('_instruct', ''))
        cross_origin_task_cs[data_args.dataset_names[i].replace("_instruct", "")] = []
        for o1 in task_prompts:
            cross_origin_task_cs[
                data_args.dataset_names[i].replace("_instruct", "")
            ].append([])
            for o2 in task_prompts:
                # print(o1, o2)
                tp1 = task_prompts[o1][i]
                tp2 = task_prompts[o2][i]

                cross_origin_task_cs[
                    data_args.dataset_names[i].replace("_instruct", "")
                ][-1].append(cosine_sim(tp1, tp2).item())

        cross_origin_task_cs[data_args.dataset_names[i].replace("_instruct", "")] = (
            torch.Tensor(
                cross_origin_task_cs[
                    data_args.dataset_names[i].replace("_instruct", "")
                ]
            )
        )

    return cross_origin_task_cs


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

parser = ArgumentParser(
    (TrainingArguments, DataTrainingArguments, PromptArithmeticsConfig)
)

training_args, data_args, pa_config = parser.parse_toml_file(
    "configs/cross_origin.toml"
)

# training_args, data_args, pa_config = parser.parse_toml_file(
#     "configs/prompt_tuning/single-task/llama31_8b_instruct.toml"
# )
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
print(dup_tpv_ct_cs)

tpv_ct_cs = dict(
    filter(
        lambda x: x[0] in remove_duplicates(dup_tpv_ct_cs.keys()), dup_tpv_ct_cs.items()
    )
)

print(f"tpv_ct_cs_{timestamp}")
# print(tpv_ct_cs)
create_heatmaps(
    tpv_ct_cs,
    filename_prefix=f"tpv_ct_cs_{timestamp}",
    save_dir=f"./visuals/{timestamp}",
    n_rows=11,
    figsize=(25, 40),
)

tpv_ct_cs = dup_tpv_ct_cs
avg_ct_co_tpv_mean = defaultdict(lambda: defaultdict(dict))
avg_ct_co_tpv_std = defaultdict(lambda: defaultdict(dict))
print("average cross origin, cross tasks, task prompt vector cosine similarity:")
for dataset_name in tpv_ct_cs:
    n = len(tpv_ct_cs[dataset_name.replace("_instruct", "")])

    if (
        mapping[dataset_name.replace("_instruct", "")].split(" ")[0]
        == mapping[dataset_name.replace("_instruct", "")].split(" ")[1]
    ):
        no_diag = tpv_ct_cs[dataset_name.replace("_instruct", "")].masked_select(
            torch.tril(torch.ones(n, n, dtype=bool), diagonal=-1)
        )
    else:
        no_diag = tpv_ct_cs[dataset_name.replace("_instruct", "")]

    # print(no_diag.shape)
    # print(dataset_name.replace('_instruct', ''), np.round(no_diag.mean().item(), 2), np.round(no_diag.std().item(), 2))
    avg_ct_co_tpv_mean[mapping[dataset_name.replace("_instruct", "")].split(" ")[0]][
        mapping[dataset_name.replace("_instruct", "")].split(" ")[1]
    ] = np.round(no_diag.mean().item(), 2)
    avg_ct_co_tpv_std[mapping[dataset_name.replace("_instruct", "")].split(" ")[0]][
        mapping[dataset_name.replace("_instruct", "")].split(" ")[1]
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
    n_rows=11,
    figsize=(25, 40),
)

task_ct_cs = dup_task_ct_cs
avg_ct_co_task_mean = defaultdict(lambda: defaultdict(dict))
avg_ct_co_task_std = defaultdict(lambda: defaultdict(dict))
print("average cross origin, cross tasks, task prompt cosine similarity:")
for dataset_name in task_ct_cs:
    n = len(task_ct_cs[dataset_name.replace("_instruct", "")])
    if (
        mapping[dataset_name.replace("_instruct", "")].split(" ")[0]
        == mapping[dataset_name.replace("_instruct", "")].split(" ")[1]
    ):
        no_diag = task_ct_cs[dataset_name.replace("_instruct", "")].masked_select(
            torch.tril(torch.ones(n, n, dtype=bool), diagonal=-1)
        )
    else:
        no_diag = tpv_ct_cs[dataset_name.replace("_instruct", "")]

    # print(no_diag)
    # print(dataset_name.replace('_instruct', ''), np.round(no_diag.mean().item(), 2), np.round(no_diag.std().item(), 2))
    avg_ct_co_task_mean[mapping[dataset_name.replace("_instruct", "")].split(" ")[0]][
        mapping[dataset_name.replace("_instruct", "")].split(" ")[1]
    ] = np.round(no_diag.mean().item(), 2)
    avg_ct_co_task_std[mapping[dataset_name.replace("_instruct", "")].split(" ")[0]][
        mapping[dataset_name.replace("_instruct", "")].split(" ")[1]
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
    n = len(cross_origin_task_cs[dataset_name.replace("_instruct", "")])
    no_diag = (
        cross_origin_task_cs[dataset_name.replace("_instruct", "")]
        .masked_select(~torch.eye(n, dtype=bool))
        .view(n, n - 1)
    )
    print(
        dataset_name.replace("_instruct", ""),
        np.round(no_diag.mean().item(), 2),
        np.round(no_diag.std().item(), 2),
    )

print("average cross origin task prompt vector cosine similarity:")
cross_origin_tpv_cs = get_tpv_comparison(
    data_args, task_prompt_vectors, name_func_map["cosine"]
)
for dataset_name in cross_origin_tpv_cs:
    n = len(cross_origin_tpv_cs[dataset_name.replace("_instruct", "")])
    no_diag = (
        cross_origin_tpv_cs[dataset_name.replace("_instruct", "")]
        .masked_select(~torch.eye(n, dtype=bool))
        .view(n, n - 1)
    )
    print(
        dataset_name.replace("_instruct", ""),
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


#         if not os.path.exists(f"./results/cross_origin/{timestamp}"):
#             os.makedirs(f"./results/cross_origin/{timestamp}")

#         results.to_csv(f"./results/cross_origin/{timestamp}/results_{o1}_{o2}_{timestamp}.csv")
