import seaborn as sns

from typing import List, Dict

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np

mapping = {
    "dbpedia_text_dbpedia_text": "DBPedia/DBPedia",
    "dbpedia_text": "DBPedia",
    "dbpedia_text_mnli_text": "DBPedia/MNLI",
    "dbpedia_text_qnli_text": "DBPedia/QNLI",
    "dbpedia_text_sst2_text": "DBPedia/SST2",
    "dbpedia_text_trec_coarse_text": "DBPedia/TREC",
    "dbpedia_text_yelp_polarity_text": "DBPedia/Yelp",
    "dbpedia_text_math": "DBPedia/MATH",
    "dbpedia_text_squad_v2": "DBPedia/SQuADv2",
    "mnli_text_dbpedia_text": "MNLI/DBPedia",
    "mnli_text_mnli_text": "MNLI/MNLI",
    "mnli_text": "MNLI",
    "mnli_text_qnli_text": "MNLI/QNLI",
    "mnli_text_sst2_text": "MNLI/SST2",
    "mnli_text_trec_coarse_text": "MNLI/TREC",
    "mnli_text_yelp_polarity_text": "MNLI/Yelp",
    "mnli_text_math": "MNLI/MATH",
    "mnli_text_squad_v2": "MNLI/SQuADv2",
    "qnli_text_dbpedia_text": "QNLI/DBPedia",
    "qnli_text_mnli_text": "QNLI/MNLI",
    "qnli_text_qnli_text": "QNLI/QNLI",
    "qnli_text": "QNLI",
    "qnli_text_sst2_text": "QNLI/SST2",
    "qnli_text_trec_coarse_text": "QNLI/TREC",
    "qnli_text_yelp_polarity_text": "QNLI/Yelp",
    "qnli_text_math": "QNLI/MATH",
    "qnli_text_squad_v2": "QNLI/SQuADv2",
    "sst2_text_dbpedia_text": "SST2/DBPedia",
    "sst2_text_mnli_text": "SST2/MNLI",
    "sst2_text_qnli_text": "SST2/QNLI",
    "sst2_text_sst2_text": "SST2/SST2",
    "sst2_text": "SST2",
    "sst2_text_trec_coarse_text": "SST2/TREC",
    "sst2_text_yelp_polarity_text": "SST2/Yelp",
    "sst2_text_math": "SST2/MATH",
    "sst2_text_squad_v2": "SST2/SQuADv2",
    "trec_coarse_text_dbpedia_text": "TREC/DBPedia",
    "trec_coarse_text_mnli_text": "TREC/MNLI",
    "trec_coarse_text_qnli_text": "TREC/QNLI",
    "trec_coarse_text_sst2_text": "TREC/SST2",
    "trec_coarse_text_trec_coarse_text": "TREC/TREC",
    "trec_coarse_text": "TREC",
    "trec_coarse_text_yelp_polarity_text": "TREC/Yelp",
    "trec_coarse_text_math": "TREC/MATH",
    "trec_coarse_text_squad_v2": "TREC/SQuADv2",
    "yelp_polarity_text_dbpedia_text": "Yelp/DBPedia",
    "yelp_polarity_text_mnli_text": "Yelp/MNLI",
    "yelp_polarity_text_qnli_text": "Yelp/QNLI",
    "yelp_polarity_text_sst2_text": "Yelp/SST2",
    "yelp_polarity_text_trec_coarse_text": "Yelp/TREC",
    "yelp_polarity_text_yelp_polarity_text": "Yelp/Yelp",
    "yelp_polarity_text_math": "Yelp/MATH",
    "yelp_polarity_text_squad_v2": "Yelp/SQuADv2",
    "yelp_polarity_text": "Yelp",
    "math": "MATH",
    "squad_v2": "SQuADv2",
    "math_dbpedia_text": "MATH/DBPedia",
    "math_mnli_text": "MATH/MNLI",
    "math_qnli_text": "MATH/QNLI",
    "math_sst2_text": "MATH/SST2",
    "math_trec_coarse_text": "MATH/TREC",
    "math_yelp_polarity_text": "MATH/Yelp",
    "squad_v2_dbpedia_text": "SQuADv2/DBPedia",
    "squad_v2_mnli_text": "SQuADv2/MNLI",
    "squad_v2_qnli_text": "SQuADv2/QNLI",
    "squad_v2_sst2_text": "SQuADv2/SST2",
    "squad_v2_trec_coarse_text": "SQuADv2/TREC",
    "squad_v2_yelp_polarity_text": "SQuADv2/Yelp",
    "math_math": "MATH/MATH",
    "squad_v2_squad_v2": "SQuADv2/SQuADv2",
    "math_squad_v2": "MATH/SQuADv2",
    "squad_v2_math": "SQuADv2/MATH",
}


def create_heatmaps(
    data: Dict[str, List[List[float]]],
    filename_prefix: str = "",
    n_rows=2,
    save_dir="./visuals",
    figsize=(35, 15),
    save_pages=1,
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_cols = round(len(data.values()) / n_rows)
    # for name in data:

    #     df = pd.DataFrame(data[name])

    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(
    #         df, annot=True, fmt=".2f", cmap="crest", mask=np.eye(len(data[name]))
    #     )
    #     plt.savefig(
    #         f"{save_dir}/{filename_prefix}_{name}_heatmap.png", bbox_inches="tight"
    #     )
    #     plt.close()

    print(n_rows, n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i in range(n_rows):
        for j in range(n_cols):

            # print(list(data.values())[i * n_cols + j])
            if (
                len(mapping[list(data.keys())[i * n_cols + j]].split("/")) != 1
                and mapping[list(data.keys())[i * n_cols + j]].split("/")[0]
                == mapping[list(data.keys())[i * n_cols + j]].split("/")[1]
            ):
                df = pd.DataFrame(list(data.values())[i * n_cols + j][1:, :-1])
                # print(df)
            else:
                df = pd.DataFrame(list(data.values())[i * n_cols + j])
                # print(df)

            if len(mapping[list(data.keys())[i * n_cols + j]].split("/")) == 1:
                axs[i, j].set_title(mapping[list(data.keys())[i * n_cols + j]])

            mask = None

            if len(mapping[list(data.keys())[i * n_cols + j]].split("/")) == 1:
                mask = np.triu(np.ones((len(df.axes[0]), len(df.axes[0])), dtype=bool))
            elif (
                mapping[list(data.keys())[i * n_cols + j]].split("/")[0]
                == mapping[list(data.keys())[i * n_cols + j]].split("/")[1]
            ):
                mask = np.invert(
                    np.tril(np.ones((len(df.axes[0]), len(df.axes[0])), dtype=bool))
                )

            ax = sns.heatmap(
                df,
                annot=True,
                fmt=".2f",
                cmap="crest",
                ax=axs[i, j],
                mask=mask,
            )

            if len(mapping[list(data.keys())[i * n_cols + j]].split("/")) != 1:
                ax.set_ylabel(mapping[list(data.keys())[i * n_cols + j]].split("/")[0])
                ax.set_xlabel(mapping[list(data.keys())[i * n_cols + j]].split("/")[1])

            if len(mapping[list(data.keys())[i * n_cols + j]].split("/")) == 1:
                ticks = ax.get_xticks()
                ax.set_xticks(ticks[:-1])

                ticks = ax.get_yticks()
                ax.set_yticks(ticks[1:])

            if (
                len(mapping[list(data.keys())[i * n_cols + j]].split("/")) != 1
                and mapping[list(data.keys())[i * n_cols + j]].split("/")[0]
                == mapping[list(data.keys())[i * n_cols + j]].split("/")[1]
            ):

                labels = ax.get_yticklabels()

                # print(labels)
                new_labels = np.arange(len(labels)) + 1

                for k, nl in enumerate(new_labels):
                    labels[k] = str(nl)

                ax.set_yticklabels(labels)
                # print(ax.get_yticklabels())

            ax.invert_yaxis()

    fig.tight_layout()
    plt.savefig(f"{save_dir}/{filename_prefix}_heatmaps.png", bbox_inches="tight")
    plt.savefig(f"{save_dir}/{filename_prefix}_heatmaps.pdf", bbox_inches="tight")

    plt.close()


def create_plots(
    data,
    filename_prefix,
    n_rows,
    save_dir="./visuals",
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
