import argparse

import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

task_labels = {
    "dbpedia_text mnli_text": "DBPedia + MNLI",
    "dbpedia_text qnli_text": "DBPedia + QNLI",
    "dbpedia_text sst2_text": "DBPedia + SST2",
    "dbpedia_text trec_coarse_text": "DBPedia + TREC",
    "dbpedia_text yelp_polarity_text": "DBPedia + Yelp",
    "mnli_text qnli_text": "MNLI + QNLI",
    "mnli_text sst2_text": "MNLI + SST2",
    "mnli_text trec_coarse_text": "MNLI + TREC",
    "mnli_text yelp_polarity_text": "MNLI + Yelp",
    "qnli_text sst2_text": "QNLI + SST2",
    "qnli_text trec_coarse_text": "QNLI + TREC",
    "qnli_text yelp_polarity_text": "QNLI + Yelp",
    "sst2_text trec_coarse_text": "SST2 + TREC",
    "sst2_text yelp_polarity_text": "SST2 + Yelp",
    "trec_coarse_text yelp_polarity_text": "TREC + Yelp",
}

bold_labels = [
    "DBPedia + MNLI",
    "DBPedia + QNLI",
    "MNLI + QNLI",
    "MNLI + Yelp",
    "QNLI + SST2",
    "QNLI + Yelp",
    "SST2 + Yelp",
]

# task_labels = {
#     "dbpedia_text mnli_text": "DBPedia + MNLI",
#     "dbpedia_text qnli_text": "DBPedia + QNLI",
#     "mnli_text qnli_text": "MNLI + QNLI",
#     "mnli_text yelp_polarity_text": "MNLI + Yelp",
#     "qnli_text sst2_text": "QNLI + SST2",
#     "qnli_text yelp_polarity_text": "QNLI + Yelp",
#     "sst2_text yelp_polarity_text": "SST2 + Yelp",
# }


argparse_parser = argparse.ArgumentParser(
    prog="Create figures from results",
    description="Compute average and std over all results.",
)

argparse_parser.add_argument("results_dir", help="Path to the results directory.")
args = argparse_parser.parse_args()

df = pd.read_csv(f"{args.results_dir}/data_results_10.csv", index_col=0)

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


font = {"size": 32}

matplotlib.rc("font", **font)
# Plotting the updated bar plot
plt.figure(figsize=(30, 18))
ax = sns.barplot(data=df_melted, x="tasks", y="Score", hue="Task Type", palette="tab10")
# ax.set(ylim=[0.6,1])
plt.ylabel(None)
plt.xlabel(None)
plt.xticks(rotation=90)
plt.legend(title="Task Type", loc="lower left")
plt.grid(True, axis="y")


ax = plt.gca()
for lab in ax.get_xticklabels():
    if lab.get_text() in bold_labels:
        lab.set_fontweight("bold")

plt.savefig("combinations.png", bbox_inches="tight")
plt.savefig("combinations.pdf", bbox_inches="tight")
