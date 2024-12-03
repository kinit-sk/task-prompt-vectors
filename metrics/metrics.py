from sklearn.metrics import f1_score
from metrics.utils import check_data_state, binary_reverse

import numpy as np

import evaluate


# def exact_match(preds, targets):
#     check_data_state(preds, targets)

#     metric = evaluate.load("exact_match", keep_in_memory=True)

#     return metric.compute(predictions=preds, references=targets)

def squad_v2_metric(preds, targets):
    metric = evaluate.load("squad_v2")
    
    return metric.compute(predictions=preds, references=targets)

def exact_match(preds, targets):
    check_data_state(preds, targets)

    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    # print(preds, targets)
    return {"exact_match": np.sum(preds == targets) / preds.size}


def f1(preds, targets, labels):
    check_data_state(preds, targets)

    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

    return {"f1": f1_score(targets, preds, labels=labels, pos_label=labels[1])}


def macro_f1(preds, targets, labels):
    check_data_state(preds, targets)

    return {"macro_f1": f1_score(targets, preds, labels=labels, average="macro")}
