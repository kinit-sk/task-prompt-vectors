import numpy as np
import torch

from .utils import binary_reverse, string_to_float

import evaluate

eval_f1: evaluate.EvaluationModule = evaluate.load("f1")
eval_accuracy: evaluate.EvaluationModule = evaluate.load("accuracy")
eval_pearsonr = evaluate.load("pearsonr")
eval_spearmanr = evaluate.load("spearmanr")


def f1_score_with_invalid(preds, targets):
    assert len(preds) == len(targets)

    preds, targets = np.asarray(preds), np.asarray(targets)

    invalid_idx_mask = np.logical_and(preds != "0", preds != "1")
    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])

    preds, targets = torch.tensor(preds.astype(np.int32)), torch.tensor(
        targets.astype(np.int32)
    )

    return eval_f1.compute(predictions=preds, references=targets)


def accuracy_with_invalid(preds, targets):
    assert len(preds) == len(targets)

    preds, targets = np.asarray(preds), np.asarray(targets)

    invalid_idx_mask = np.logical_and(preds != "0", preds != "1")
    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])

    preds, targets = torch.tensor(preds.astype(np.int32)), torch.tensor(
        targets.astype(np.int32)
    )

    return eval_accuracy.compute(predictions=preds, references=targets)


def pearsonr(preds, targets):
    assert len(preds) == len(targets)

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return eval_pearsonr.compute(predictions=preds, references=targets)


def spearmanr(preds, targets):
    assert len(preds) == len(targets)

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return eval_spearmanr.compute(predictions=preds, references=targets)
