import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

import glob

import argparse

from collections import defaultdict


argparse_parser = argparse.ArgumentParser(
    prog="Create figures from results",
    description="Compute average and std over all results.",
)

argparse_parser.add_argument("results_dir", help="Path to the results directory.")
args = argparse_parser.parse_args()

dfs = []

for file in sorted(glob.glob(f"{args.results_dir}results_origin*.csv")):

    # print(file)
    df = pd.read_csv(file, index_col=0)

    df["tasks"] = df["tasks"].map(lambda x: " ".join(sorted(x.split(" "))))

    dfs.append(df)

accuracy_per_task = defaultdict(lambda: defaultdict(list))
for df in dfs:
    for t in df["tasks"]:
        for tt in t.split(" "):
            accuracy_per_task[t][tt].append(df[df["tasks"] == t][f"{tt}_accuracy"].values[0])

print(accuracy_per_task)


for t in accuracy_per_task:
    boxplot_dict = defaultdict(list)

    t_split = t.split(" ")
    
    if len(t_split) > 1:
        for tt in t_split:
            boxplot_dict["tasks"] += [tt]*len(accuracy_per_task[tt][tt])
            boxplot_dict["accuracy"] += accuracy_per_task[tt][tt]

            boxplot_dict["tasks"] += [f"add({tt})"]*len(accuracy_per_task[t][tt])
            boxplot_dict["accuracy"] += accuracy_per_task[t][tt]

        bdf = pd.DataFrame.from_dict(boxplot_dict)
        # print(bdf)

        boxplot = sns.boxplot(bdf, y="accuracy", x="tasks")
        fig = boxplot.get_figure()
        boxplot.set_title(t)
        fig.savefig(f"./visuals/results_box_{t}.png", bbox_inches="tight")
        
        plt.close()

exit()

mean_dfs = pd.concat(dfs).groupby("tasks", as_index=False).mean()
mean_dfs_std = pd.concat(dfs).groupby("tasks", as_index=False).std()

mean_dfs.to_csv("average_10.csv")
mean_dfs_std.to_csv("std_10.csv")

data_dfs = []

for df in [mean_dfs, mean_dfs_std]:
    res_dict = {}

    for t in df["tasks"]:
        print(t)
        for tt in t.split(" "):
            if t not in res_dict:
                res_dict[t] = {}

            res_dict[t].update(
                {
                    tt: df[df["tasks"] == t][f"{tt}_accuracy"].values[0]
                    / mean_dfs[df["tasks"] == tt][f"{tt}_accuracy"].values[0]
                }
            )

    print(res_dict)

    data = {"first_task": [], "second_task": [], "tasks": []}

    for t in res_dict:
        if len(t.split(" ")) == 2:
            first_task, second_task = t.split(" ")
            data["first_task"].append(res_dict[t][first_task])
            data["second_task"].append(res_dict[t][second_task])
            data["tasks"].append(t)

    data = pd.DataFrame.from_dict(data)

    print(data)
    data_dfs.append(data)

scatterplot = sns.scatterplot(
    data=data_dfs[0], x="first_task", y="second_task", hue="tasks", style="tasks"
)

plt.vlines(x=1, ymin=0, ymax=1.2, colors="gray", linestyles="dashed")
plt.hlines(y=1, xmin=0, xmax=1.2, colors="gray", linestyles="dashed")
scatterplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))

fig = scatterplot.get_figure()
fig.savefig("results_10.png", bbox_inches="tight")

data_dfs[0].to_csv("data_results_10.csv")
data_dfs[1].to_csv("data_results_std_10.csv")