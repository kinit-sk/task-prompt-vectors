import seaborn as sns

from typing import List, Dict

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

save_dir = "./visuals"


def create_heatmaps(
    data: Dict[str, List[List[float]]], filename_prefix: str = "", n_rows=2
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_cols = len(data.values()) // n_rows
    for name in data:
        # print(name)

        df = pd.DataFrame(data[name])
        # print(df)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="crest")
        plt.savefig(
            f"{save_dir}/{filename_prefix}_{name}_heatmap.png", bbox_inches="tight"
        )
        plt.close()

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(35, 15))

    for i in range(n_rows):
        for j in range(n_cols):

            df = pd.DataFrame(list(data.values())[i * n_cols + j])
            # print(list(data.keys())[i*n_cols+j], end=" ")

            axs[i, j].set_title(list(data.keys())[i * n_cols + j])

            sns.heatmap(df, annot=True, fmt=".2f", cmap="crest", ax=axs[i, j])

        # print()

    plt.savefig(f"{save_dir}/{filename_prefix}_heatmaps.png", bbox_inches="tight")

    plt.close()


def create_plots(data, filename_prefix, n_rows) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
