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
    "rte_text": "RTE",
    "rte_text_dbpedia_text": "RTE/DBPedia",
    "rte_text_mnli_text": "RTE/MNLI",
    "rte_text_qnli_text": "RTE/QNLI",
    "rte_text_sst2_text": "RTE/SST2",
    "rte_text_trec_coarse_text": "RTE/TREC",
    "rte_text_yelp_polarity_text": "RTE/Yelp",
    "rte_text_math": "RTE/MATH",
    "rte_text_squad_v2": "RTE/SQuADv2",
    "rte_text_rte_text": "RTE/RTE",
    "rte_text_stsb_text": "RTE/STS-B",
    "rte_text_mrpc_text": "RTE/MRPC",
    "rte_text_cola_text": "RTE/CoLA",
    "rte_text_qqp_text": "RTE/QQP",
    "stsb_text": "STS-B",
    "stsb_text_dbpedia_text": "STS-B/DBPedia",
    "stsb_text_mnli_text": "STS-B/MNLI",
    "stsb_text_qnli_text": "STS-B/QNLI",
    "stsb_text_sst2_text": "STS-B/SST2",
    "stsb_text_trec_coarse_text": "STS-B/TREC",
    "stsb_text_yelp_polarity_text": "STS-B/Yelp",
    "stsb_text_math": "STS-B/MATH",
    "stsb_text_squad_v2": "STS-B/SQuADv2",
    "stsb_text_rte_text": "STS-B/RTE",
    "stsb_text_stsb_text": "STS-B/STS-B",
    "stsb_text_mrpc_text": "STS-B/MRPC",
    "stsb_text_cola_text": "STS-B/CoLA",
    "stsb_text_qqp_text": "STS-B/QQP",
    "mrpc_text": "MRPC",
    "mrpc_text_dbpedia_text": "MRPC/DBPedia",
    "mrpc_text_mnli_text": "MRPC/MNLI",
    "mrpc_text_qnli_text": "MRPC/QNLI",
    "mrpc_text_sst2_text": "MRPC/SST2",
    "mrpc_text_trec_coarse_text": "MRPC/TREC",
    "mrpc_text_yelp_polarity_text": "MRPC/Yelp",
    "mrpc_text_math": "MRPC/MATH",
    "mrpc_text_squad_v2": "MRPC/SQuADv2",
    "mrpc_text_rte_text": "MRPC/RTE",
    "mrpc_text_stsb_text": "MRPC/STS-B",
    "mrpc_text_mrpc_text": "MRPC/MRPC",
    "mrpc_text_cola_text": "MRPC/CoLA",
    "mrpc_text_qqp_text": "MRPC/QQP",
    "cola_text": "CoLA",
    "cola_text_dbpedia_text": "CoLA/DBPedia",
    "cola_text_mnli_text": "CoLA/MNLI",
    "cola_text_qnli_text": "CoLA/QNLI",
    "cola_text_sst2_text": "CoLA/SST2",
    "cola_text_trec_coarse_text": "CoLA/TREC",
    "cola_text_yelp_polarity_text": "CoLA/Yelp",
    "cola_text_math": "CoLA/MATH",
    "cola_text_squad_v2": "CoLA/SQuADv2",
    "cola_text_rte_text": "CoLA/RTE",
    "cola_text_stsb_text": "CoLA/STS-B",
    "cola_text_mrpc_text": "CoLA/MRPC",
    "cola_text_cola_text": "CoLA/CoLA",
    "cola_text_qqp_text": "CoLA/QQP",
    "qqp_text": "QQP",
    "qqp_text_dbpedia_text": "QQP/DBPedia",
    "qqp_text_mnli_text": "QQP/MNLI",
    "qqp_text_qnli_text": "QQP/QNLI",
    "qqp_text_sst2_text": "QQP/SST2",
    "qqp_text_trec_coarse_text": "QQP/TREC",
    "qqp_text_yelp_polarity_text": "QQP/Yelp",
    "qqp_text_math": "QQP/MATH",
    "qqp_text_squad_v2": "QQP/SQuADv2",
    "qqp_text_rte_text": "QQP/RTE",
    "qqp_text_stsb_text": "QQP/STS-B",
    "qqp_text_mrpc_text": "QQP/MRPC",
    "qqp_text_cola_text": "QQP/CoLA",
    "qqp_text_qqp_text": "QQP/QQP",
    "dbpedia_text_rte_text": "DBPedia/RTE",
    "mnli_text_rte_text": "MNLI/RTE",
    "qnli_text_rte_text": "QNLI/RTE",
    "sst2_text_rte_text": "SST2/RTE",
    "trec_coarse_text_rte_text": "TREC/RTE",
    "yelp_polarity_text_rte_text": "Yelp/RTE",
    "math_rte_text": "MATH/RTE",
    "squad_v2_rte_text": "SQuADv2/RTE",
    "rte_text_rte_text": "RTE/RTE",
    "stsb_text_rte_text": "STS-B/RTE",
    "mrpc_text_rte_text": "MRPC/RTE",
    "cola_text_rte_text": "CoLA/RTE",
    "qqp_text_rte_text": "QQP/RTE",
    "dbpedia_text_stsb_text": "DBPedia/STS-B",
    "mnli_text_stsb_text": "MNLI/STS-B",
    "qnli_text_stsb_text": "QNLI/STS-B",
    "sst2_text_stsb_text": "SST2/STS-B",
    "trec_coarse_text_stsb_text": "TREC/STS-B",
    "yelp_polarity_text_stsb_text": "Yelp/STS-B",
    "math_stsb_text": "MATH/STS-B",
    "squad_v2_stsb_text": "SQuADv2/STS-B",
    "rte_text_stsb_text": "RTE/STS-B",
    "stsb_text_stsb_text": "STS-B/STS-B",
    "mrpc_text_stsb_text": "MRPC/STS-B",
    "cola_text_stsb_text": "CoLA/STS-B",
    "qqp_text_stsb_text": "QQP/STS-B",
    "dbpedia_text_cola_text": "DBPedia/CoLA",
    "mnli_text_cola_text": "MNLI/CoLA",
    "qnli_text_cola_text": "QNLI/CoLA",
    "sst2_text_cola_text": "SST2/CoLA",
    "trec_coarse_text_cola_text": "TREC/CoLA",
    "yelp_polarity_text_cola_text": "Yelp/CoLA",
    "math_cola_text": "MATH/CoLA",
    "squad_v2_cola_text": "SQuADv2/CoLA",
    "rte_text_cola_text": "RTE/CoLA",
    "stsb_text_cola_text": "STS-B/CoLA",
    "mrpc_text_cola_text": "MRPC/CoLA",
    "cola_text_cola_text": "CoLA/CoLA",
    "qqp_text_cola_text": "QQP/CoLA",
    "dbpedia_text_mrpc_text": "DBPedia/MRPC",
    "mnli_text_mrpc_text": "MNLI/MRPC",
    "qnli_text_mrpc_text": "QNLI/MRPC",
    "sst2_text_mrpc_text": "SST2/MRPC",
    "trec_coarse_text_mrpc_text": "TREC/MRPC",
    "yelp_polarity_text_mrpc_text": "Yelp/MRPC",
    "math_mrpc_text": "MATH/MRPC",
    "squad_v2_mrpc_text": "SQuADv2/MRPC",
    "rte_text_mrpc_text": "RTE/MRPC",
    "stsb_text_mrpc_text": "STS-B/MRPC",
    "mrpc_text_mrpc_text": "MRPC/MRPC",
    "cola_text_mrpc_text": "CoLA/MRPC",
    "qqp_text_mrpc_text": "QQP/MRPC",
    "dbpedia_text_qqp_text": "DBPedia/QQP",
    "mnli_text_qqp_text": "MNLI/QQP",
    "qnli_text_qqp_text": "QNLI/QQP",
    "sst2_text_qqp_text": "SST2/QQP",
    "trec_coarse_text_qqp_text": "TREC/QQP",
    "yelp_polarity_text_qqp_text": "Yelp/QQP",
    "math_qqp_text": "MATH/QQP",
    "squad_v2_qqp_text": "SQuADv2/QQP",
    "rte_text_qqp_text": "RTE/QQP",
    "stsb_text_qqp_text": "STS-B/QQP",
    "mrpc_text_qqp_text": "MRPC/QQP",
    "cola_text_qqp_text": "CoLA/QQP",
    "qqp_text_qqp_text": "QQP/QQP",
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

    print(len(data.values()))
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

            if (i * n_cols + j) >= len(data.values()):
                continue

            # print(i * n_cols + j)
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
