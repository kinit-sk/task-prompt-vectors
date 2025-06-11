# import wandb

# timestamps = ["04262024123207"]

# api = wandb.Api()

# query = {"config.run_name": {"$regex": "04262024123207"}}
# runs = api.runs("rbelanec/eval_arithmetics", filters=query)
# print(len(runs))

# for r in runs:
#     print(r.history(keys=["accuracy"]))


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

import sys

metrics = ["f1", "exact_match"]

tasks = ["nli", "cls", "sent"]


test_datasets = {
    "nli": ["scitail_text", "snli_text"],
    "sent": ["imdb_text", "sst5_text"],
    "cls": ["ag_news_text", "yahoo_text"],
}

train_datasets = {
    "nli": [
        "origin_.?.",
        "mnli_text",
        "qnli_text",
        "mnli_text_qnli_text_spot",
        "mnli_text_qnli_text_attempt",
        "mnli_text_qnli_text",
    ],
    "sent": [
        "origin_.?.",
        "sst2_text",
        "yelp_polarity_text",
        "sst2_text_yelp_polarity_text_spot",
        "sst2_text_yelp_polarity_text_attempt",
        "sst2_text_yelp_polarity_text",
    ],
    "cls": [
        "origin_.?.",
        "dbpedia_text",
        "trec_coarse_text",
        "dbpedia_text_trec_coarse_text_spot",
        "dbpedia_text_trec_coarse_text_attempt",
        "dbpedia_text_trec_coarse_text",
    ],
}

formating_map = {
    "origin_.?.": "random",
    "dbpedia_text": "DBPedia (SPoT)",
    "trec_coarse_text": "TREC Coarse (SPoT)",
    "dbpedia_text_trec_coarse_text": "DBPedia + TREC Coarse (Ours)",
    "dbpedia_text_trec_coarse_text_spot": "DBPedia + TREC Coarse (SPoT)",
    "dbpedia_text_trec_coarse_text_attempt": "DBPedia + TREC Coarse (ATTEMPT)",
    "sst2_text": "SST2 (SPoT)",
    "yelp_polarity_text": "Yelp (SPoT)",
    "sst2_text_yelp_polarity_text": "SST2 + Yelp (Ours)",
    "sst2_text_yelp_polarity_text_spot": "SST2 + Yelp (SPoT)",
    "sst2_text_yelp_polarity_text_attempt": "SST2 + Yelp (ATTEMPT)",
    "mnli_text": "MNLI (SPoT)",
    "qnli_text": "QNLI (SPoT)",
    "mnli_text_qnli_text": "MNLI + QNLI (Ours)",
    "mnli_text_qnli_text_spot": "MNLI + QNLI (SPoT)",
    "mnli_text_qnli_text_attempt": "MNLI + QNLI (ATTEMPT)",
    "scitail_text": "SciTail",
    "snli_text": "SNLI",
    "sst5_text": "SST5",
    "imdb_text": "IMDB",
    "ag_news_text": "AG News",
    "yahoo_text": "Yahoo",
}


n_classes = {
    "scitail_text": 2,
    "snli_text": 3,
    "sst5_text": 5,
    "imdb_text": 2,
    "ag_news_text": 4,
    "yahoo_text": 10,
}

font = {"size": 50}
matplotlib.rc("font", **font)


fig, axs = plt.subplots(nrows=len(tasks), ncols=2, figsize=(45, 48))
# fig, axs = plt.subplots(nrows=1, ncols=len(tasks), figsize=(23,8))

legend_handles = {}

for ti, t in enumerate(tasks):
    df = pd.read_csv(f"wandb_results_{t}.csv", index_col=0)
    for i, td in enumerate(test_datasets[t]):
        for d in train_datasets[t]:

            data = df[df.index.str.contains(rf".*_{td}_{d}$")][1:7]
            # print(data)

            metric = data["f1"].values
            std = data["f1_std"].values

            shots = [int(i.split("_")[0]) for i in data.index]

            print(metric, std, shots)

            if (
                d == "dbpedia_text_trec_coarse_text"
                or d == "sst2_text_yelp_polarity_text"
                or d == "mnli_text_qnli_text"
            ):
                (line,) = axs[ti, i].plot(
                    shots, metric, label=formating_map[d], marker="o", linewidth=3.0
                )
            else:
                (line,) = axs[ti, i].plot(
                    shots,
                    metric,
                    label=formating_map[d],
                    marker="o",
                    linestyle="--",
                    linewidth=3.0,
                )

            if i == 0:
                if ti not in legend_handles:
                    legend_handles[ti] = []
                legend_handles[ti].append((line, formating_map[d]))
            # axs[ti, i].fill_between(shots, metric - std, metric + std, alpha=0.2)

        axs[ti, i].set_xscale("log")

        axs[ti, i].set_xticks(shots)
        axs[ti, i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        axs[ti, i].set_title(formating_map[td])
        axs[ti, i].set_xlabel("N shots")
        axs[ti, i].set_ylabel("Macro F1")

        handles, labels = zip(*legend_handles[ti])
        fig.legend(
            handles,
            labels,
            # title="Train dataset",
            loc="center",
            bbox_to_anchor=(0.5, (len(tasks) - ti - 0.9) / len(tasks)),  # y-position by row
            bbox_transform=fig.transFigure,
            ncol=len(train_datasets),
        )

        # if i % 2 == 0:
        #     axs[ti, i].legend(
        #         loc="upper center",
        #         bbox_to_anchor=(1.0, -0.2),
        #         fancybox=True,
        #         shadow=True,
        #         ncol=3,
        #     )


plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.subplots_adjust(hspace=0.6) 

# fig.legend(loc='upper left')
plt.savefig(f"./fewshots_all.pdf")
plt.savefig(f"./fewshots_all.png")
plt.close()


test_datasets = {
    "nli": ["scitail_text"],
    "sent": ["imdb_text"],
    "cls": ["ag_news_text"],
}

# 3 figures
fig, axs = plt.subplots(nrows=1, ncols=len(tasks), figsize=(57, 18))

for ti, t in enumerate(tasks):
    df = pd.read_csv(f"wandb_results_{t}.csv", index_col=0)
    for i, td in enumerate(test_datasets[t]):
        for d in train_datasets[t]:

            data = df[df.index.str.contains(rf".*_{td}_{d}$")][1:7]
            # print(data)

            metric = data["f1"].values
            std = data["f1_std"].values

            shots = [int(i.split("_")[0]) for i in data.index]

            print(metric, std, shots)

            if (
                d == "dbpedia_text_trec_coarse_text"
                or d == "sst2_text_yelp_polarity_text"
                or d == "mnli_text_qnli_text"
            ):
                axs[ti].plot(
                    shots, metric, label=formating_map[d], marker="o", linewidth=3.5
                )
            else:
                axs[ti].plot(
                    shots,
                    metric,
                    label=formating_map[d],
                    marker="o",
                    linestyle="--",
                    linewidth=3.0,
                )
            # axs[ti].fill_between(shots, metric - std, metric + std, alpha=0.2)

        axs[ti].set_xscale("log")

        axs[ti].set_xticks(shots)

        axs[ti].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        axs[ti].set_title(formating_map[td])
        axs[ti].set_xlabel("N shots")
        axs[ti].set_ylabel("Macro F1")
        axs[ti].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=2,
        )

fig.tight_layout(pad=1.0)

# fig.legend(loc='upper left')
plt.savefig(f"./fewshots_three.pdf")
plt.savefig(f"./fewshots_three.png")
plt.close()
