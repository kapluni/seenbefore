"""
tune.py — Automatic prompt tuning with DSPy
===========================================
Optimizes the ``AntisemitismClassifier`` against the labeled dataset. Two
optimizers are supported:

  * mipro  (default) — dspy.MIPROv2. Jointly searches over *instruction
    rewrites* (proposed by the stronger prompt model) and *few-shot
    demonstrations* bootstrapped from the training data, using Bayesian
    optimization on the dev set. This is the canonical "auto prompt tuning".
  * bootstrap — dspy.BootstrapFewShotWithRandomSearch. Lighter/cheaper:
    keeps the initial instructions and only searches over demonstrations.

The optimized program (its tuned instructions + selected demonstrations) is
saved to classifier/artifacts/ as JSON and can be reloaded by evaluate.py and
predict.py — no need to re-run the (paid) optimization.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m classifier.tune                 # MIPROv2, light auto mode
    python -m classifier.tune --optimizer bootstrap
    python -m classifier.tune --auto medium --metric safety
"""

import argparse

import dspy

from .config import OPTIMIZED_PROGRAM_PATH, ARTIFACTS_DIR
from .data import load_all
from .program import AntisemitismClassifier
from .metrics import accuracy_metric, safety_metric, aggregate_report, print_report
from .lm import configure_task_lm, make_prompt_lm


METRICS = {"accuracy": accuracy_metric, "safety": safety_metric}


def _evaluate(program, examples, metric):
    """Run the program over examples and return (report, avg_metric)."""
    preds, scores = [], []
    for ex in examples:
        pred = program(text=ex.text)
        preds.append(pred)
        scores.append(float(metric(ex, pred)))
    report = aggregate_report(examples, preds)
    return report, sum(scores) / len(scores) if scores else 0.0


def tune(optimizer="mipro", auto="light", metric_name="accuracy",
         max_demos=4, num_threads=8):
    metric = METRICS[metric_name]

    task_lm = configure_task_lm()
    print(f"Task model   : {task_lm.model}")

    train, dev, test = load_all()
    print(f"Loaded {len(train)} train / {len(dev)} dev / {len(test)} test examples")

    student = AntisemitismClassifier()

    # ---- Baseline (un-tuned initial prompt) -------------------------------
    print("\nScoring baseline (initial hand-written prompt) on dev...")
    base_report, base_score = _evaluate(student, dev, metric)
    print_report("BASELINE (dev)", base_report)

    # ---- Optimize ---------------------------------------------------------
    if optimizer == "mipro":
        prompt_lm = make_prompt_lm()
        print(f"\nPrompt model : {prompt_lm.model}")
        print(f"Optimizing with MIPROv2 (auto={auto})...")
        opt = dspy.MIPROv2(
            metric=metric,
            auto=auto,
            prompt_model=prompt_lm,
            task_model=task_lm,
            num_threads=num_threads,
        )
        optimized = opt.compile(
            student,
            trainset=train,
            valset=dev,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
            requires_permission_to_run=False,
        )
    elif optimizer == "bootstrap":
        print("\nOptimizing with BootstrapFewShotWithRandomSearch...")
        opt = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
            num_candidate_programs=8,
            num_threads=num_threads,
        )
        optimized = opt.compile(student, trainset=train, valset=dev)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # ---- Compare on dev ---------------------------------------------------
    print("\nScoring optimized program on dev...")
    opt_report, opt_score = _evaluate(optimized, dev, metric)
    print_report("OPTIMIZED (dev)", opt_report)

    print(f"\nDev {metric_name} metric:  baseline {base_score:.3f}  →  optimized {opt_score:.3f}"
          f"   (Δ {opt_score - base_score:+.3f})")

    # ---- Save -------------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    optimized.save(str(OPTIMIZED_PROGRAM_PATH))
    print(f"\nSaved optimized program → {OPTIMIZED_PROGRAM_PATH}")
    print("Evaluate it on the held-out test set with:\n    python -m classifier.evaluate")
    return optimized


def main():
    p = argparse.ArgumentParser(description="Auto-tune the antisemitism classifier prompt with DSPy.")
    p.add_argument("--optimizer", choices=["mipro", "bootstrap"], default="mipro")
    p.add_argument("--auto", choices=["light", "medium", "heavy"], default="light",
                   help="MIPROv2 search budget (more = better but pricier)")
    p.add_argument("--metric", choices=list(METRICS), default="accuracy")
    p.add_argument("--max-demos", type=int, default=4,
                   help="Max few-shot demonstrations to attach")
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()

    tune(optimizer=args.optimizer, auto=args.auto, metric_name=args.metric,
         max_demos=args.max_demos, num_threads=args.num_threads)


if __name__ == "__main__":
    main()
