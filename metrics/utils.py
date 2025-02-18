import regex as re
import string
import collections
import numpy as np


def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)

def _f1_score(target, prediction):
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _exact_match_score(target, prediction):
    return target == prediction


def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
    return max(metric_fn(ground_truth, prediction) for ground_truth in ground_truths)


def _normalize_answer(text, punc_chars, punc_repl):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)
    return text


def normalize_squad(answer):
    return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def qa_metrics(targets, predictions):
    # print(targets, len(targets))
    # print(predictions, len(predictions))
    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must match.")
    em = np.mean(
        [
            _metric_max_over_ground_truths(_exact_match_score, t, p)
            for p, t in zip(predictions, targets)
        ]
    )
    f1 = np.mean(
        [
            _metric_max_over_ground_truths(_f1_score, t, p)
            for p, t in zip(predictions, targets)
        ]
    )
    em *= 100
    f1 *= 100
    return {"em": em, "f1": f1}


def binary_reverse(targets, labels):
    return [labels[0] if target == labels[1] else labels[1] for target in targets]


def string_to_float(string, default=-1.0):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def check_data_state(preds, targets):
    assert len(preds) == len(targets)
