import wandb
import numpy as np
import pandas as pd
import sys

api = wandb.Api()

shots = {
    "nli": [0, 5, 10, 25, 50, 100, 250, 500, 750, 1000],
    "sent": [0, 5, 25, 50, 100, 250, 500, 750, 1000],
    "cls": [0, 10, 25, 50, 100, 250, 500, 750, 1000],
}

test_datasets = {
    "nli": ["scitail_text", "snli_text"],
    "sent": ["sst5_text", "imdb_text"],
    "cls": ["ag_news_text", "yahoo_text"],
}

train_datasets = {
    "nli": ["origin_.?", "mnli_text", "qnli_text", "mnli_text_qnli_text"],
    "sent": [
        "origin_.?",
        "sst2_text",
        "yelp_polarity_text",
        "sst2_text_yelp_polarity_text",
    ],
    "cls": [
        "origin_.?",
        "dbpedia_text",
        "trec_coarse_text",
        "dbpedia_text_trec_coarse_text",
    ],
}

tasks = ["nli", "cls", "sent"]
# tasks = ["cls"]

results = {}

for t in tasks:
    for s in shots[t]:
        for td in test_datasets[t]:
            for d in train_datasets[t]:

                query = {
                    "config.run_name": {
                        "$regex": f"fewshot_{s}_.*.{td}.*.origin_.?._{d}.$"
                    }
                }
                # if t == "cls" and s == 50:
                #     query = {"config.run_name": {"$regex": f"fewshot_{s}_05172024204829_{td}.*.origin_.?._{d}.$"}}

                # query = {"config.run_name": {"$regex": f"fewshot_{s}.*.{td}.*.origin_.?._mnli_text.$"}}
                print(query)
                runs = api.runs("rbelanec/eval_arithmetics_fewshot", filters=query)
                print("n_runs:", len(runs))

                # if len(runs) > 10:
                #     runs = runs[:10]

                assert len(runs) == 10

                results[f"{s}_{td}_{d}"] = {}

                ems = []
                f1s = []

                for r in runs:
                    summary = r.summary

                    # print(r.history())

                    print(r.name)
                    entry = dict(
                        filter(
                            lambda x: "test" in x[0]
                            and ("f1" in x[0] or "exact_match" in x[0]),
                            summary.items(),
                        )
                    )
                    # print(list(summary.items()))

                    if "test/macro_f1" in entry.keys():
                        f1s.append(entry["test/macro_f1"])
                    elif "test/f1" in entry.keys():
                        f1s.append(entry["test/f1"])

                    ems.append(entry["test/exact_match"])

                print(ems, f1s)
                ems = np.array(ems) * 100
                f1s = np.array(f1s) * 100
                results[f"{s}_{td}_{d}"]["exact_match"] = np.round(ems.mean(), 1)
                results[f"{s}_{td}_{d}"]["exact_match_std"] = np.round(ems.std(), 1)
                results[f"{s}_{td}_{d}"]["f1"] = np.round(f1s.mean(), 1)
                results[f"{s}_{td}_{d}"]["f1_std"] = np.round(f1s.std(), 1)

                # print(results[f"{s}_{td}_{d}"])

    df = pd.DataFrame.from_dict(results).T
    df.to_csv(f"wandb_results_{t}.csv")
    print(df)
