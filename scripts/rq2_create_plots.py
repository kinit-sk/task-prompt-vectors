import argparse

import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

task_labels = {
    "dbpedia_text dbpedia_text": "DBP+DBP",
    "dbpedia_text": "DBP",
    "dbpedia_text mnli_text": "DBP+MNLI",
    "dbpedia_text qnli_text": "DBP+QNLI",
    "dbpedia_text sst2_text": "DBP+SST2",
    "dbpedia_text trec_coarse_text": "DBP+TREC",
    "dbpedia_text yelp_polarity_text": "DBP+Yelp",
    "dbpedia_text math": "DBP+MATH",
    "dbpedia_text squad_v2": "DBP+SQuADv2",
    "mnli_text dbpedia_text": "MNLI+DBP",
    "mnli_text mnli_text": "MNLI+MNLI",
    "mnli_text": "MNLI",
    "mnli_text qnli_text": "MNLI+QNLI",
    "mnli_text sst2_text": "MNLI+SST2",
    "mnli_text trec_coarse_text": "MNLI+TREC",
    "mnli_text yelp_polarity_text": "MNLI+Yelp",
    "mnli_text math": "MNLI+MATH",
    "mnli_text squad_v2": "MNLI+SQuADv2",
    "qnli_text dbpedia_text": "QNLI+DBP",
    "qnli_text mnli_text": "QNLI+MNLI",
    "qnli_text qnli_text": "QNLI+QNLI",
    "qnli_text": "QNLI",
    "qnli_text sst2_text": "QNLI+SST2",
    "qnli_text trec_coarse_text": "QNLI+TREC",
    "qnli_text yelp_polarity_text": "QNLI+Yelp",
    "qnli_text math": "QNLI+MATH",
    "qnli_text squad_v2": "QNLI+SQuADv2",
    "sst2_text dbpedia_text": "SST2+DBP",
    "sst2_text mnli_text": "SST2+MNLI",
    "sst2_text qnli_text": "SST2+QNLI",
    "sst2_text sst2_text": "SST2+SST2",
    "sst2_text": "SST2",
    "sst2_text trec_coarse_text": "SST2+TREC",
    "sst2_text yelp_polarity_text": "SST2+Yelp",
    "sst2_text math": "SST2+MATH",
    "sst2_text squad_v2": "SST2+SQuADv2",
    "trec_coarse_text dbpedia_text": "TREC+DBP",
    "trec_coarse_text mnli_text": "TREC+MNLI",
    "trec_coarse_text qnli_text": "TREC+QNLI",
    "trec_coarse_text sst2_text": "TREC+SST2",
    "trec_coarse_text trec_coarse_text": "TREC+TREC",
    "trec_coarse_text": "TREC",
    "trec_coarse_text yelp_polarity_text": "TREC+Yelp",
    "trec_coarse_text math": "TREC+MATH",
    "trec_coarse_text squad_v2": "TREC+SQuADv2",
    "yelp_polarity_text dbpedia_text": "Yelp+DBP",
    "yelp_polarity_text mnli_text": "Yelp+MNLI",
    "yelp_polarity_text qnli_text": "Yelp+QNLI",
    "yelp_polarity_text sst2_text": "Yelp+SST2",
    "yelp_polarity_text trec_coarse_text": "Yelp+TREC",
    "yelp_polarity_text yelp_polarity_text": "Yelp+Yelp",
    "yelp_polarity_text math": "Yelp+MATH",
    "yelp_polarity_text squad_v2": "Yelp+SQuADv2",
    "yelp_polarity_text": "Yelp",
    "math": "MATH",
    "squad_v2": "SQuADv2",
    "math dbpedia_text": "MATH+DBP",
    "math mnli_text": "MATH+MNLI",
    "math qnli_text": "MATH+QNLI",
    "math sst2_text": "MATH+SST2",
    "math trec_coarse_text": "MATH+TREC",
    "math yelp_polarity_text": "MATH+Yelp",
    "squad_v2_dbpedia_text": "SQuADv2+DBP",
    "squad_v2_mnli_text": "SQuADv2+MNLI",
    "squad_v2_qnli_text": "SQuADv2+QNLI",
    "squad_v2_sst2_text": "SQuADv2+SST2",
    "squad_v2_trec_coarse_text": "SQuADv2+TREC",
    "squad_v2_yelp_polarity_text": "SQuADv2+Yelp",
    "math math": "MATH+MATH",
    "squad_v2_squad_v2": "SQuADv2+SQuADv2",
    "math squad_v2": "MATH+SQuADv2",
    "squad_v2_math": "SQuADv2+MATH",
    "rte_text": "RTE",
    "rte_text dbpedia_text": "RTE+DBP",
    "rte_text mnli_text": "RTE+MNLI",
    "rte_text qnli_text": "RTE+QNLI",
    "rte_text sst2_text": "RTE+SST2",
    "rte_text trec_coarse_text": "RTE+TREC",
    "rte_text yelp_polarity_text": "RTE+Yelp",
    "rte_text math": "RTE+MATH",
    "rte_text squad_v2": "RTE+SQuADv2",
    "rte_text rte_text": "RTE+RTE",
    "rte_text stsb_text": "RTE+STS-B",
    "rte_text mrpc_text": "RTE+MRPC",
    "rte_text cola_text": "RTE+CoLA",
    "rte_text qqp_text": "RTE+QQP",
    "stsb_text": "STS-B",
    "stsb_text dbpedia_text": "STS-B+DBP",
    "stsb_text mnli_text": "STS-B+MNLI",
    "stsb_text qnli_text": "STS-B+QNLI",
    "stsb_text sst2_text": "STS-B+SST2",
    "stsb_text trec_coarse_text": "STS-B+TREC",
    "stsb_text yelp_polarity_text": "STS-B+Yelp",
    "stsb_text math": "STS-B+MATH",
    "stsb_text squad_v2": "STS-B+SQuADv2",
    "stsb_text rte_text": "STS-B+RTE",
    "stsb_text stsb_text": "STS-B+STS-B",
    "stsb_text mrpc_text": "STS-B+MRPC",
    "stsb_text cola_text": "STS-B+CoLA",
    "stsb_text qqp_text": "STS-B+QQP",
    "mrpc_text": "MRPC",
    "mrpc_text dbpedia_text": "MRPC+DBP",
    "mrpc_text mnli_text": "MRPC+MNLI",
    "mrpc_text qnli_text": "MRPC+QNLI",
    "mrpc_text sst2_text": "MRPC+SST2",
    "mrpc_text trec_coarse_text": "MRPC+TREC",
    "mrpc_text yelp_polarity_text": "MRPC+Yelp",
    "mrpc_text math": "MRPC+MATH",
    "mrpc_text squad_v2": "MRPC+SQuADv2",
    "mrpc_text rte_text": "MRPC+RTE",
    "mrpc_text stsb_text": "MRPC+STS-B",
    "mrpc_text mrpc_text": "MRPC+MRPC",
    "mrpc_text cola_text": "MRPC+CoLA",
    "mrpc_text qqp_text": "MRPC+QQP",
    "cola_text": "CoLA",
    "cola_text dbpedia_text": "CoLA+DBP",
    "cola_text mnli_text": "CoLA+MNLI",
    "cola_text qnli_text": "CoLA+QNLI",
    "cola_text sst2_text": "CoLA+SST2",
    "cola_text trec_coarse_text": "CoLA+TREC",
    "cola_text yelp_polarity_text": "CoLA+Yelp",
    "cola_text math": "CoLA+MATH",
    "cola_text squad_v2": "CoLA+SQuADv2",
    "cola_text rte_text": "CoLA+RTE",
    "cola_text stsb_text": "CoLA+STS-B",
    "cola_text mrpc_text": "CoLA+MRPC",
    "cola_text cola_text": "CoLA+CoLA",
    "cola_text qqp_text": "CoLA+QQP",
    "qqp_text": "QQP",
    "qqp_text dbpedia_text": "QQP+DBP",
    "qqp_text mnli_text": "QQP+MNLI",
    "qqp_text qnli_text": "QQP+QNLI",
    "qqp_text sst2_text": "QQP+SST2",
    "qqp_text trec_coarse_text": "QQP+TREC",
    "qqp_text yelp_polarity_text": "QQP+Yelp",
    "qqp_text math": "QQP+MATH",
    "qqp_text squad_v2": "QQP+SQuADv2",
    "qqp_text rte_text": "QQP+RTE",
    "qqp_text stsb_text": "QQP+STS-B",
    "qqp_text mrpc_text": "QQP+MRPC",
    "qqp_text cola_text": "QQP+CoLA",
    "qqp_text qqp_text": "QQP+QQP",
    "dbpedia_text rte_text": "DBP+RTE",
    "mnli_text rte_text": "MNLI+RTE",
    "qnli_text rte_text": "QNLI+RTE",
    "sst2_text rte_text": "SST2+RTE",
    "trec_coarse_text rte_text": "TREC+RTE",
    "yelp_polarity_text rte_text": "Yelp+RTE",
    "math rte_text": "MATH+RTE",
    "squad_v2_rte_text": "SQuADv2+RTE",
    "rte_text rte_text": "RTE+RTE",
    "stsb_text rte_text": "STS-B+RTE",
    "mrpc_text rte_text": "MRPC+RTE",
    "cola_text rte_text": "CoLA+RTE",
    "qqp_text rte_text": "QQP+RTE",
    "dbpedia_text stsb_text": "DBP+STS-B",
    "mnli_text stsb_text": "MNLI+STS-B",
    "qnli_text stsb_text": "QNLI+STS-B",
    "sst2_text stsb_text": "SST2+STS-B",
    "trec_coarse_text stsb_text": "TREC+STS-B",
    "yelp_polarity_text stsb_text": "Yelp+STS-B",
    "math stsb_text": "MATH+STS-B",
    "squad_v2_stsb_text": "SQuADv2+STS-B",
    "rte_text stsb_text": "RTE+STS-B",
    "stsb_text stsb_text": "STS-B+STS-B",
    "mrpc_text stsb_text": "MRPC+STS-B",
    "cola_text stsb_text": "CoLA+STS-B",
    "qqp_text stsb_text": "QQP+STS-B",
    "dbpedia_text cola_text": "DBP+CoLA",
    "mnli_text cola_text": "MNLI+CoLA",
    "qnli_text cola_text": "QNLI+CoLA",
    "sst2_text cola_text": "SST2+CoLA",
    "trec_coarse_text cola_text": "TREC+CoLA",
    "yelp_polarity_text cola_text": "Yelp+CoLA",
    "math cola_text": "MATH+CoLA",
    "squad_v2_cola_text": "SQuADv2+CoLA",
    "rte_text cola_text": "RTE+CoLA",
    "stsb_text cola_text": "STS-B+CoLA",
    "mrpc_text cola_text": "MRPC+CoLA",
    "cola_text cola_text": "CoLA+CoLA",
    "qqp_text cola_text": "QQP+CoLA",
    "dbpedia_text mrpc_text": "DBP+MRPC",
    "mnli_text mrpc_text": "MNLI+MRPC",
    "qnli_text mrpc_text": "QNLI+MRPC",
    "sst2_text mrpc_text": "SST2+MRPC",
    "trec_coarse_text mrpc_text": "TREC+MRPC",
    "yelp_polarity_text mrpc_text": "Yelp+MRPC",
    "math mrpc_text": "MATH+MRPC",
    "squad_v2_mrpc_text": "SQuADv2+MRPC",
    "rte_text mrpc_text": "RTE+MRPC",
    "stsb_text mrpc_text": "STS-B+MRPC",
    "mrpc_text mrpc_text": "MRPC+MRPC",
    "cola_text mrpc_text": "CoLA+MRPC",
    "qqp_text mrpc_text": "QQP+MRPC",
    "dbpedia_text qqp_text": "DBP+QQP",
    "mnli_text qqp_text": "MNLI+QQP",
    "qnli_text qqp_text": "QNLI+QQP",
    "sst2_text qqp_text": "SST2+QQP",
    "trec_coarse_text qqp_text": "TREC+QQP",
    "yelp_polarity_text qqp_text": "Yelp+QQP",
    "math qqp_text": "MATH+QQP",
    "squad_v2_qqp_text": "SQuADv2+QQP",
    "rte_text qqp_text": "RTE+QQP",
    "stsb_text qqp_text": "STS-B+QQP",
    "mrpc_text qqp_text": "MRPC+QQP",
    "cola_text qqp_text": "CoLA+QQP",
    "qqp_text qqp_text": "QQP+QQP",
}

bold_labels = [
    # "DBP+MNLI",
    # "DBP+QNLI",
    "MNLI+QNLI",
    "MNLI+Yelp",
    "QNLI+SST2",
    "QNLI+Yelp",
    "SST2+Yelp",
    "CoLA+MRPC",
    "CoLA+QQP",
    "CoLA+RTE",
    "CoLA+SST2",
    "CoLA+STS-B",
    "CoLA+TREC",
    "CoLA+QNLI",
    "DBP+QQP",
    "MNLI+MRPC",
    "MNLI+QQP",
    "MNLI+RTE",
    "MRPC+QNLI",
    "MRPC+QQP",
    "MRPC+RTE",
    "MRPC+SST2",
    "MRPC+STS-B",
    "MRPC+TREC",
    "QNLI+QQP",
    "QNLI+RTE",
    "QNLI+STS-B",
    "QQP+RTE",
    "QQP+SST2",
    "QQP+STS-B",
    "QQP+Yelp",
    "RTE+SST2",
    "RTE+STS-B",
    "RTE+TREC",
    "SST2+STS-B",
    "STS-B+TREC",
    "CoLA+Yelp",
    "DPB+MNLI",
    "DPB+MRPC",
    "DBP+QNLI",
    "DBP+STS-B",
    "MNLI+STSB-B",
    "MNLI+SST2",
    "RTE+Yelp",

]

# task_labels = {
#     "dbpedia_text mnli_text": "DBP+MNLI",
#     "dbpedia_text qnli_text": "DBP+QNLI",
#     "mnli_text qnli_text": "MNLI+QNLI",
#     "mnli_text yelp_polarity_text": "MNLI+Yelp",
#     "qnli_text sst2_text": "QNLI+SST2",
#     "qnli_text yelp_polarity_text": "QNLI+Yelp",
#     "sst2_text yelp_polarity_text": "SST2+Yelp",
# }


argparse_parser = argparse.ArgumentParser(
    prog="Create figures from results",
    description="Compute average and std over all results.",
)

argparse_parser.add_argument("results_dir", help="Path to the results directory.")
args = argparse_parser.parse_args()

df = pd.read_csv(f"{args.results_dir}/data_results_10.csv", index_col=0)

print(df)

df["tasks"] = df["tasks"].map(task_labels)
df_melted = df.melt(
    id_vars="tasks",
    value_vars=["first_task", "second_task"],
    var_name="Task Type",
    value_name="Score",
)
df_melted["Task Type"] = df_melted["Task Type"].replace(
    {"first_task": "First Task", "second_task": "Second Task"}
)

print(df_melted.sort_values("tasks").iloc[56:])


font = {"size": 45}
matplotlib.rc("font", **font)

# figsize=(42, 18)
figsize=(52, 27)

fig, axs = plt.subplots(nrows=2, figsize=figsize)
# Plotting the updated bar plot

sns.barplot(data=df_melted.sort_values("tasks").iloc[:56], x="tasks", y="Score", hue="Task Type", palette="tab10", ax=axs[0])
plt.xticks(rotation=60)

sns.barplot(data=df_melted.sort_values("tasks").iloc[56:], x="tasks", y="Score", hue="Task Type", palette="tab10", ax=axs[1])

# ax.set(ylim=[0.6,1])
# plt.ylabel(None)
# plt.xlabel(None)
# plt.xticks(rotation=60)
# plt.legend(title="Task Type", loc="lower left")
# plt.grid(True, axis="y")

axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.55),
          fancybox=True, shadow=True, ncol=5)

axs[0].get_legend().remove()

for ax in axs:
    ax.grid(True, axis="y")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis='x', labelrotation=60)
    for lab in ax.get_xticklabels():
        if lab.get_text() in bold_labels:
            lab.set_fontweight("bold")

fig.tight_layout()
fig.supylabel("Performance relative to prompt tuning", x=-0.02, size=60)


plt.savefig("combinations.png", bbox_inches="tight")
plt.savefig("combinations.pdf", bbox_inches="tight")
