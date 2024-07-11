from scipy.stats import ttest_ind_from_stats
import pprint

import json

# Defining the data structure to hold the table data
data = {
    "SciTail (NLI)": {
        "Random": {
            "0 shots": {"F1": 54.9, "std": 6.6},
            "100 shots": {"F1": 75.6, "std": 0.5},
        },
        "MNLI (SPoT)": {
            "0 shots": {"F1": 70.4, "std": 0.4},
            "100 shots": {"F1": 87.8, "std": 0.9},
        },
        "QNLI (SPoT)": {
            "0 shots": {"F1": 57.7, "std": 13.1},
            "100 shots": {"F1": 77.7, "std": 1.3},
        },
        "QNLI + MNLI (SPoT)": {
            "0 shots": {"F1": 70.4, "std": 1.2},
            "100 shots": {"F1": 87.7, "std": 0.6},
        },
        "QNLI + MNLI (ATTEMPT)": {
            "0 shots": {"F1": 63.8, "std": 4.2},
            "100 shots": {"F1": 83.6, "std": 3.0},
        },
        "QNLI + MNLI (ours)": {
            "0 shots": {"F1": 71.5, "std": 0.8},
            "100 shots": {"F1": 88.1, "std": 0.9},
        },
    },
    "AG News (Classification)": {
        "Random": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 50.4, "std": 11.2},
        },
        "DBPedia (SPoT)": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 83.4, "std": 0.6},
        },
        "TREC (SPoT)": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 65.7, "std": 5.6},
        },
        "DBPedia + TREC (SPoT)": {
            "0 shots": {"F1": None, "std": None},
            "100 shots": {"F1": None, "std": None},
        },
        "DBPedia + TREC (ATTEMPT)": {
            "0 shots": {"F1": 11.5, "std": 1.7},
            "100 shots": {"F1": 20.7, "std": 2.8},
        },
        "DBPedia + TREC (ours)": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 83.0, "std": 0.9},
        },
    },
    "IMDB (Sentiment)": {
        "Random": {
            "0 shots": {"F1": 77.2, "std": 9.6},
            "100 shots": {"F1": 89.4, "std": 0.4},
        },
        "SST2 (SPoT)": {
            "0 shots": {"F1": 88.0, "std": 0.6},
            "100 shots": {"F1": 90.2, "std": 0.3},
        },
        "Yelp (SPoT)": {
            "0 shots": {"F1": 90.0, "std": 0.3},
            "100 shots": {"F1": 90.3, "std": 0.2},
        },
        "SST2 + Yelp (SPoT)": {
            "0 shots": {"F1": 90.8, "std": 0.2},
            "100 shots": {"F1": 90.8, "std": 0.2},
        },
        "SST2 + Yelp (ATTEMPT)": {
            "0 shots": {"F1": 79.2, "std": 6.0},
            "100 shots": {"F1": 89.4, "std": 0.8},
        },
        "SST2 + Yelp (ours)": {
            "0 shots": {"F1": 90.1, "std": 0.5},
            "100 shots": {"F1": 90.4, "std": 0.2},
        },
    },
    "SNLI (NLI)": {
        "Random": {
            "0 shots": {"F1": 46.5, "std": 1.5},
            "100 shots": {"F1": 47.6, "std": 1.9},
        },
        "MNLI (SPoT)": {
            "0 shots": {"F1": 79.5, "std": 0.3},
            "100 shots": {"F1": 80.8, "std": 0.4},
        },
        "QNLI (SPoT)": {
            "0 shots": {"F1": 47.1, "std": 0.3},
            "100 shots": {"F1": 49.1, "std": 0.9},
        },
        "QNLI + MNLI (SPoT)": {
            "0 shots": {"F1": 79.6, "std": 0.2},
            "100 shots": {"F1": 81.0, "std": 0.4},
        },
        "QNLI + MNLI (ATTEMPT)": {
            "0 shots": {"F1": 78.5, "std": 0.5},
            "100 shots": {"F1": 79.6, "std": 1.6},
        },
        "QNLI + MNLI (ours)": {
            "0 shots": {"F1": 79.2, "std": 1.4},
            "100 shots": {"F1": 80.3, "std": 0.3},
        },
    },
    "Yahoo Answers (Classification)": {
        "Random": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 27.6, "std": 10.6},
        },
        "DBPedia (SPoT)": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 61.3, "std": 1.1},
        },
        "TREC (SPoT)": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 36.5, "std": 8.7},
        },
        "DBPedia + TREC (SPoT)": {
            "0 shots": {"F1": None, "std": None},
            "100 shots": {"F1": None, "std": None},
        },
        "DBPedia + TREC (ATTEMPT)": {
            "0 shots": {"F1": 0.1, "std": 0.0},
            "100 shots": {"F1": 8.1, "std": 5.6},
        },
        "DBPedia + TREC (ours)": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 61.1, "std": 0.9},
        },
    },
    "SST5 (Sentiment)": {
        "Random": {
            "0 shots": {"F1": 0.0, "std": 0.0},
            "100 shots": {"F1": 83.2, "std": 5.8},
        },
        "SST2 (SPoT)": {
            "0 shots": {"F1": 94.0, "std": 0.3},
            "100 shots": {"F1": 93.9, "std": 0.3},
        },
        "Yelp (SPoT)": {
            "0 shots": {"F1": 88.6, "std": 0.8},
            "100 shots": {"F1": 90.6, "std": 0.5},
        },
        "SST2 + Yelp (SPoT)": {
            "0 shots": {"F1": 93.7, "std": 0.5},
            "100 shots": {"F1": 93.8, "std": 0.5},
        },
        "SST2 + Yelp (ATTEMPT)": {
            "0 shots": {"F1": 16.4, "std": 4.5},
            "100 shots": {"F1": 37.8, "std": 7.0},
        },
        "SST2 + Yelp (ours)": {
            "0 shots": {"F1": 89.9, "std": 0.8},
            "100 shots": {"F1": 91.5, "std": 0.5},
        },
    },
}

n = 10


def compute_significance(mean1, std1, n1, mean2, std2, n2):
    t_stat, p_value = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2)
    return p_value


# pprint.pprint(data)

comparisons = [
    ("SciTail (NLI)", "QNLI + MNLI (ours)", "MNLI (SPoT)"),
    ("SNLI (NLI)", "QNLI + MNLI (ours)", "MNLI (SPoT)"),
    ("AG News (Classification)", "DBPedia + TREC (ours)", "DBPedia (SPoT)"),
    ("Yahoo Answers (Classification)", "DBPedia + TREC (ours)", "DBPedia (SPoT)"),
    ("IMDB (Sentiment)", "SST2 + Yelp (ours)", "Yelp (SPoT)"),
    ("SST5 (Sentiment)", "SST2 + Yelp (ours)", "SST2 (SPoT)"),
]

comparison_results = {}

for dataset, task1, task2 in comparisons:
    comparison_results[dataset] = {}
    for shots in ["0 shots", "100 shots"]:
        mean1 = data[dataset][task1][shots]["F1"]
        std1 = data[dataset][task1][shots]["std"]
        mean2 = data[dataset][task2][shots]["F1"]
        std2 = data[dataset][task2][shots]["std"]

        p_value = compute_significance(mean1, std1, n, mean2, std2, n)

        comparison_results[dataset][f"{task1} vs {task2} ({shots})"] = {
            "task1": {"F1": mean1, "std": std1},
            "task2": {"F1": mean2, "std": std2},
            "p-value": p_value,
            "significance": bool(p_value <= 0.05 / len(comparisons)),
        }

# Printing the comparison results
pprint.pprint(comparison_results)

with open("significance.json", "w") as f:
    json.dump(comparison_results, f, indent=4)


data = {
    "dataset": ["QNLI", "MNLI", "TREC Coarse", "DBpedia", "SST2", "Yelp", "avg"],
    "Prompt tuning": {
        "QNLI": {"score": 93.3, "std": 0},
        "MNLI": {"score": 85.4, "std": 0.1},
        "TREC Coarse": {"score": 95.5, "std": 1.7},
        "DBpedia": {"score": 99.1, "std": 0},
        "SST2": {"score": 93.8, "std": 0.3},
        "Yelp": {"score": 97.2, "std": 0},
        "avg": {"score": 93.8, "std": 0.08},
    },
    "Random init": {
        "QNLI": {"score": 93.2, "std": 0.1},
        "MNLI": {"score": 85.3, "std": 0.2},
        "TREC Coarse": {"score": 26.5, "std": 18.2},
        "DBpedia": {"score": 99, "std": 0.1},
        "SST2": {"score": 93.2, "std": 0.6},
        "Yelp": {"score": 97.1, "std": 0.1},
        "avg": {"score": 93.6, "std": 0.26},
    },
}

comparison_results = {}
for dataset in data["dataset"]:
    comparison_results[dataset] = {}
    mean1 = data["Prompt tuning"][dataset]["score"]
    std1 = data["Prompt tuning"][dataset]["std"]
    mean2 = data["Random init"][dataset]["score"]
    std2 = data["Random init"][dataset]["std"]

    p_value = compute_significance(mean1, std1, n, mean2, std2, n)

    comparison_results[dataset][f"Prompt tuning vs Random init ({dataset})"] = {
        "Prompt tuning": {"score": mean1, "std": std1},
        "Random init": {"score": mean2, "std": std2},
        "p-value": p_value,
        "significance": bool(p_value <= 0.05 / len(data["dataset"])),
    }

pprint.pprint(comparison_results)

with open("significance2.json", "w") as f:
    json.dump(comparison_results, f, indent=4)
