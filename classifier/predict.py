"""
predict.py — Classify new text with the (tuned) classifier
==========================================================
Loads the optimized program if it exists (falling back to the baseline
prompt otherwise) and classifies text from the command line, a file, or
stdin. Also importable as a library:

    from classifier.predict import load_classifier
    clf = load_classifier()
    result = clf("Zionists control the media and the banks.")
    print(result["label"], "—", result["rationale"])

CLI:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m classifier.predict "some social media post"
    echo "one post per line" | python -m classifier.predict --stdin
    python -m classifier.predict --file posts.txt
"""

import sys
import json
import argparse

from .config import OPTIMIZED_PROGRAM_PATH
from .program import AntisemitismClassifier
from .lm import configure_task_lm


def load_classifier():
    """Return a callable ``clf(text) -> {label, rationale}``.

    Uses the auto-tuned program when available, else the initial prompt.
    """
    configure_task_lm()
    program = AntisemitismClassifier()
    used = "baseline (initial prompt)"
    if OPTIMIZED_PROGRAM_PATH.exists():
        program.load(str(OPTIMIZED_PROGRAM_PATH))
        used = "optimized (auto-tuned prompt)"

    def clf(text):
        pred = program(text=text)
        return {"label": pred.label, "rationale": pred.rationale}

    clf.variant = used  # type: ignore[attr-defined]
    return clf


def main():
    p = argparse.ArgumentParser(description="Classify social-media text for antisemitism.")
    p.add_argument("text", nargs="*", help="Text to classify")
    p.add_argument("--stdin", action="store_true", help="Read one post per line from stdin")
    p.add_argument("--file", help="Read one post per line from a file")
    p.add_argument("--json", action="store_true", help="Emit JSON lines")
    args = p.parse_args()

    if args.file:
        texts = [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
    elif args.stdin:
        texts = [l.strip() for l in sys.stdin if l.strip()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        p.error("provide text, --file PATH, or --stdin")

    clf = load_classifier()
    print(f"# using {clf.variant}\n", file=sys.stderr)

    for text in texts:
        result = clf(text)
        if args.json:
            print(json.dumps({"text": text, **result}, ensure_ascii=False))
        else:
            print(f"[{result['label']}] {text}")
            print(f"    ↳ {result['rationale']}\n")


if __name__ == "__main__":
    main()
