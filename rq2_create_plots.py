import argparse

import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

task_labels = {
    "dbpedia_text mnli_text": "DBPedia\nMNLI",
    "dbpedia_text qnli_text": "DBPedia\nQNLI",
    "dbpedia_text sst2_text": "DBPedia\nSST2",
    "dbpedia_text trec_coarse_text": "DBPedia\nTREC",
    "dbpedia_text yelp_polarity_text": "DBPedia\nYelp",
    "mnli_text qnli_text": "MNLI\nQNLI",
    "mnli_text sst2_text": "MNLI\nSST2",
    "mnli_text trec_coarse_text": "MNLI\nTREC",
    "mnli_text yelp_polarity_text": "MNLI\nYelp",
    "qnli_text sst2_text": "QNLI\nSST2",
    "qnli_text trec_coarse_text": "QNLI\nTREC",
    "qnli_text yelp_polarity_text": "QNLI\nYelp",
    "sst2_text trec_coarse_text": "SST2\nTREC",
    "sst2_text yelp_polarity_text": "SST2\nYelp",
    "trec_coarse_text yelp_polarity_text": "TREC\nYelp",
}

bold_labels = [
    "DBPedia\nMNLI",
    "DBPedia\nQNLI",
    "MNLI\nQNLI",
    "MNLI\nYelp",
    "QNLI\nSST2",
    "QNLI\nYelp",
    "SST2\nYelp",
]

# task_labels = {
#     "dbpedia_text mnli_text": "DBPedia\nMNLI",
#     "dbpedia_text qnli_text": "DBPedia\nQNLI",
#     "mnli_text qnli_text": "MNLI\nQNLI",
#     "mnli_text yelp_polarity_text": "MNLI\nYelp",
#     "qnli_text sst2_text": "QNLI\nSST2",
#     "qnli_text yelp_polarity_text": "QNLI\nYelp",
#     "sst2_text yelp_polarity_text": "SST2\nYelp",
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
plt.figure(figsize=(42, 18))
ax = sns.barplot(data=df_melted, x="tasks", y="Score", hue="Task Type", palette="tab10")
# ax.set(ylim=[0.6,1])
plt.ylabel(None)
plt.xlabel(None)
# plt.xticks(rotation=90)
plt.legend(title="Task Type", loc="lower left")
plt.grid(True, axis="y")


ax = plt.gca()
for lab in ax.get_xticklabels():
    if lab.get_text() in bold_labels:
        lab.set_fontweight("bold")

plt.savefig("combinations.png", bbox_inches="tight")
plt.savefig("combinations.pdf", bbox_inches="tight")
