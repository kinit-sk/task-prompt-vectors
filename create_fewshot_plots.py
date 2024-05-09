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

data_dict = {
    "origin": {
        "n_samples": [0, 100, 250],
        "average_accuracy": [0, 0, 44.6],
        "std": [0, 0, 2.7],
    },
    "mnli": {
        "n_samples": [0, 100, 250],
        "average_accuracy": [71.5, 76.53, 79.65],
        "std": [0.85, 0.7, 0.55],
    },
    "qnli": {
        "n_samples": [0, 100, 250],
        "average_accuracy": [48.5, 51.4, 54.3],
        "std": [1.55, 0.9, 1.85],
    },
    "mnli+qnli": {
        "n_samples": [0, 100, 250],
        "average_accuracy": [74.65, 77.975, 79.1],
        "std": [2.55, 0.9, 0.6],
    },
}


for d in data_dict:
    print(pd.DataFrame.from_dict(data_dict[d]))
    x = np.array(data_dict[d]["n_samples"])
    y = np.array(data_dict[d]["average_accuracy"])
    error = np.array(data_dict[d]["std"])

    print(x)

    plt.plot(x, y, label=d, marker="o")
    plt.fill_between(x, y - error, y + error, alpha=0.2)

plt.legend()
plt.savefig("visuals/fewshot_mnli_qnli.png")
