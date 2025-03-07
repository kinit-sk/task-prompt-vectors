from args import TrainingArguments, DataTrainingArguments, ArgumentParser
from arithmetics import PromptArithmeticsConfig
from utils import get_task_prompt_vectors

import torch


import numpy as np

from datetime import datetime

from torch.nn.functional import cosine_similarity


from collections import defaultdict

import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns


task_labels = {
    "dbpedia_text": "DBPedia",
    "mnli_text": "MNLI",
    "qnli_text": "QNLI",
    "sst2_text": "SST2",
    "trec_coarse_text": "TREC",
    "yelp_polarity_text": "Yelp",
    "dbpedia_text_instruct": "DBPedia",
    "mnli_text_instruct": "MNLI",
    "qnli_text_instruct": "QNLI",
    "sst2_text_instruct": "SST2",
    "trec_coarse_text_instruct": "TREC",
    "yelp_polarity_text_instruct": "Yelp",
    "math_instruct": "MATH",
    "squad_v2_instruct": "SQuADv2",
    "rte_text": "RTE",
    "mrpc_text": "MRPC",
    "cola_text": "CoLA",
    "qqp_text": "QQP",
    "stsb_text": "STS-B",
}


def average_diff(t1, t2):
    return torch.abs(t1 - t2).mean()


def l2_norm(t1, t2):
    return torch.abs(torch.norm(t1) - torch.norm(t2))


def cosine_sim(t1, t2):
    return cosine_similarity(t1.flatten(), t2.flatten(), dim=0)


def get_tpv_cross_task_cs(task_prompt_vectors):
    cross_task_cs = defaultdict(lambda: defaultdict(dict))

    for origin in task_prompt_vectors:
        for tpv1 in task_prompt_vectors[origin]:
            for tpv2 in task_prompt_vectors[origin]:
                cross_task_cs[origin][list(tpv1.tasks)[0]][list(tpv2.tasks)[0]] = (
                    cosine_sim(tpv1.prompt, tpv2.prompt).item()
                )

    return cross_task_cs


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

cross_task_cs = get_tpv_cross_task_cs(task_prompt_vectors)

average_similarities = defaultdict(lambda: defaultdict(float))
counts = defaultdict(lambda: defaultdict(int))

for origin, datasets in cross_task_cs.items():
    for dataset, similarities in datasets.items():
        for compared_dataset, score in similarities.items():
            average_similarities[dataset][compared_dataset] += score
            counts[dataset][compared_dataset] += 1

for dataset, similarities in average_similarities.items():
    for compared_dataset in similarities:
        average_similarities[dataset][compared_dataset] /= counts[dataset][
            compared_dataset
        ]
        # average_similarities[dataset][compared_dataset] *= 100
        average_similarities[dataset][compared_dataset] = np.round(
            average_similarities[dataset][compared_dataset], 3
        )


df = pd.DataFrame.from_dict(average_similarities)
df = df.rename(task_labels)
df = df.rename(task_labels, axis="columns")

# df = df[["MNLI", "QNLI", "DBPedia", "TREC", "SST2", "Yelp", "MATH", "SQuADv2"]].reindex(
#     ["MNLI", "QNLI", "DBPedia", "TREC", "SST2", "Yelp", "MATH", "SQuADv2"]
# )

df = df[
    [
        "MNLI",
        "QQP",
        "QNLI",
        "SST2",
        "STS-B",
        "MRPC",
        "RTE",
        "CoLA",
        "TREC",
        "DBPedia",
        "Yelp",
    ]
].reindex(
    [
        "MNLI",
        "QQP",
        "QNLI",
        "SST2",
        "STS-B",
        "MRPC",
        "RTE",
        "CoLA",
        "TREC",
        "DBPedia",
        "Yelp",
    ]
)

# plt.figure(figsize=(16,10))
mask = np.triu(df.to_numpy())
ax = sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="crest",
    mask=mask,
)
ticks = ax.get_xticks()
ax.set_xticks(ticks[:-1])

ticks = ax.get_yticks()
ax.set_yticks(ticks[1:])

ax.tick_params(
    axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, labelrotation=90
)


# plt.title('Cross Task Comparison of Task Prompt Vectors')
plt.savefig(f"rq1_heatmap.png", bbox_inches="tight")
plt.savefig(f"rq1_heatmap.pdf", bbox_inches="tight")

plt.close()

# order = ["MNLI", "QNLI", "DBPedia", "TREC", "SST2", "Yelp", "MATH", "SQuADv2"]
order = [
    "MNLI",
    "QQP",
    "QNLI",
    "SST2",
    "STS-B",
    "MRPC",
    "RTE",
    "CoLA",
    "TREC",
    "DBPedia",
    "Yelp",
]

df_mean = pd.read_csv("avg_ct_co_tpv_mean.csv", index_col=0).loc[order, order]
df_std = pd.read_csv("avg_ct_co_tpv_std.csv", index_col=0).loc[order, order]

print(df_mean)

ax = sns.heatmap(
    df_mean,
    annot=True,
    fmt=".2f",
    cmap="crest",
    mask=np.invert(np.tril(np.ones((len(df.axes[0]), len(df.axes[0])), dtype=bool))),
)

ax.invert_yaxis()
# ax.invert_xaxis()

# ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, labelrotation=90)

# plt.title('Cross Task Comparison of Task Prompt Vectors')
plt.savefig(f"rq1_heatmap_tpv_cs.png", bbox_inches="tight")
plt.savefig(f"rq1_heatmap_tpv_cs.pdf", bbox_inches="tight")

plt.close()

df_mean = pd.read_csv("avg_ct_co_task_mean.csv", index_col=0).loc[order, order]
df_std = pd.read_csv("avg_ct_co_task_std.csv", index_col=0).loc[order, order]

print(df_mean)

ax = sns.heatmap(
    df_mean,
    annot=False,
    fmt=".2f",
    cmap="crest",
    mask=np.invert(np.tril(np.ones((len(df.axes[0]), len(df.axes[0])), dtype=bool))),
)
ax.invert_yaxis()
# ax.invert_xaxis()

# ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, labelrotation=90)

# ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# plt.title('Cross Task Comparison of Task Prompts')
plt.savefig(f"rq1_heatmap_tp_cs.png", bbox_inches="tight")
plt.savefig(f"rq1_heatmap_tp_cs.pdf", bbox_inches="tight")

plt.close()
