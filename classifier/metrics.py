"""
metrics.py — Optimization metric and evaluation reporting
=========================================================
Two things live here:

1. Per-example metrics the DSPy optimizer maximizes. They follow the DSPy
   convention ``metric(gold, pred, trace=None)`` and return a bool during
   bootstrapping (``trace`` is set) and a float during scoring.

2. ``aggregate_report`` — full classification metrics (precision / recall /
   F1 / confusion matrix) plus the project-specific number that matters most:
   the false-positive rate, i.e. how often legitimate criticism gets wrongly
   flagged as antisemitic. The project's credibility depends on keeping that low.
"""

from .config import LABEL_POS, LABEL_NEG
from .program import normalize_label


def _labels(gold, pred):
    g = normalize_label(getattr(gold, "label", gold))
    p = normalize_label(getattr(pred, "label", pred))
    return g, p


def accuracy_metric(gold, pred, trace=None):
    """Exact-match accuracy. The simple, interpretable default."""
    g, p = _labels(gold, pred)
    return g == p


def safety_metric(gold, pred, trace=None):
    """Credibility-weighted metric.

    A false positive (flagging legitimate criticism as antisemitic) is the
    most damaging error for this project, so it scores 0. A false negative
    (missing real antisemitism) is bad but less reputationally fatal, so it
    earns partial credit. This nudges the optimizer toward precision on the
    positive class without ignoring recall.

    During bootstrapping (``trace`` set) only exactly-correct predictions
    qualify as demonstrations, so we return a strict bool there.
    """
    g, p = _labels(gold, pred)
    correct = g == p
    if trace is not None:
        return correct
    if correct:
        return 1.0
    # Incorrect: distinguish the two error types.
    if g == LABEL_NEG and p == LABEL_POS:   # false positive — worst case
        return 0.0
    return 0.4                               # false negative — partial credit


def aggregate_report(golds, preds):
    """Compute full metrics over parallel lists of gold/pred labels or objects."""
    tp = tn = fp = fn = 0
    for gold, pred in zip(golds, preds):
        g, p = _labels(gold, pred)
        if g == LABEL_POS and p == LABEL_POS:
            tp += 1
        elif g == LABEL_NEG and p == LABEL_NEG:
            tn += 1
        elif g == LABEL_NEG and p == LABEL_POS:
            fp += 1
        elif g == LABEL_POS and p == LABEL_NEG:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # Of all genuinely legitimate posts, how many did we wrongly flag?
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "n": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp_rate, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def print_report(title, report):
    """Pretty-print a report dict to stdout."""
    print(f"\n=== {title} ===")
    print(f"  examples : {report['n']}")
    print(f"  accuracy : {report['accuracy']:.3f}")
    print(f"  precision: {report['precision']:.3f}  (of posts flagged, how many were truly antisemitic)")
    print(f"  recall   : {report['recall']:.3f}  (of truly antisemitic posts, how many we caught)")
    print(f"  F1       : {report['f1']:.3f}")
    print(f"  FP rate  : {report['false_positive_rate']:.3f}  (legitimate criticism wrongly flagged — keep LOW)")
    c = report["confusion"]
    print(f"  confusion: TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
