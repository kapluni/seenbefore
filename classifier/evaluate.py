"""
evaluate.py — Score the classifier on the held-out test set
===========================================================
Reports full metrics for the baseline (initial prompt) and, if present, the
tuned program saved by tune.py — so you can see exactly what auto prompt
tuning bought you on data neither was optimized against.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m classifier.evaluate              # baseline + optimized (if saved)
    python -m classifier.evaluate --baseline-only
"""

import argparse

from .config import OPTIMIZED_PROGRAM_PATH, TEST_PATH
from .data import load_split
from .program import AntisemitismClassifier
from .metrics import aggregate_report, print_report
from .lm import configure_task_lm


def _run(program, examples):
    preds = [program(text=ex.text) for ex in examples]
    return aggregate_report(examples, preds)


def main():
    p = argparse.ArgumentParser(description="Evaluate the classifier on the test set.")
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args()

    configure_task_lm()
    test = load_split(TEST_PATH)
    print(f"Loaded {len(test)} test examples")

    baseline = AntisemitismClassifier()
    print_report("BASELINE — initial prompt (test)", _run(baseline, test))

    if not args.baseline_only:
        if OPTIMIZED_PROGRAM_PATH.exists():
            tuned = AntisemitismClassifier()
            tuned.load(str(OPTIMIZED_PROGRAM_PATH))
            print_report("OPTIMIZED — auto-tuned prompt (test)", _run(tuned, test))
        else:
            print(f"\n(no optimized program at {OPTIMIZED_PROGRAM_PATH} — run "
                  f"`python -m classifier.tune` first)")


if __name__ == "__main__":
    main()
