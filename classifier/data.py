"""
data.py — Build and load the labeled classification dataset
===========================================================
Turns the raw downloaded datasets (ISCA tweets, optionally CONAN) into
balanced train/dev/test splits of (text, label) pairs, then exposes them
as ``dspy.Example`` lists for optimization and evaluation.

Why this matters for credibility
---------------------------------
Every negative example in the ISCA data already mentions Jews / Israel /
Zionism (it was surfaced by those search keywords but human-labeled
"not biased"). That makes the negatives *hard negatives*: legitimate
criticism of Israel, news reporting, neutral mentions. Training and tuning
against these — rather than random off-topic tweets — is what teaches the
classifier the line the project cares most about: antisemitism vs.
legitimate criticism of Israeli policy.

Build the dataset (no API key needed):

    python -m classifier.data --build

Outputs JSONL files in classifier/data/. Each row: {"text": ..., "label": ...}
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path

from .config import (
    RAW_SOURCES_DIR, DATA_DIR, TRAIN_PATH, DEV_PATH, TEST_PATH,
    LABEL_POS, LABEL_NEG, RANDOM_SEED,
)

# Reuse the battle-tested tweet cleaner from the existing ETL so the
# classifier sees text formatted the same way as the rest of the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from process_modern_sources import clean_tweet_text  # type: ignore
except Exception:  # pragma: no cover - fallback if import path changes
    import re

    def clean_tweet_text(text):
        t = text.replace("\\n", " ").replace("\n", " ")
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(r"@\w+", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

csv.field_size_limit(10_000_000)

# Raw ISCA CSVs and how to read them. All three share the Biased column.
ISCA_FILES = [
    (RAW_SOURCES_DIR / "isca_zenodo" / "data.csv", "isca_zenodo"),
    (RAW_SOURCES_DIR / "isca_classdata" / "ClassData2022and2023.csv", "isca_classdata"),
    (RAW_SOURCES_DIR / "isca_huggingface" / "DatasetForMachineLearning.csv", "isca_huggingface"),
]

CONAN_MULTITARGET = RAW_SOURCES_DIR / "conan" / "Multitarget-CONAN.csv"

# Minimum cleaned length to keep an example — drops fragments that carry no
# classifiable signal (e.g. a lone "@user lol").
MIN_LEN = 25


def _find_col(fieldnames, candidates):
    """Case-insensitive column lookup."""
    lowered = {c.lower().strip(): c for c in (fieldnames or [])}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _read_isca(path, source):
    """Yield (text, label, source) from an ISCA CSV using the Biased column."""
    if not path.exists():
        print(f"  ⚠ Not found, skipping: {path}")
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        text_col = _find_col(reader.fieldnames, ("text", "tweet", "content"))
        bias_col = _find_col(reader.fieldnames, ("biased", "label", "antisemitic"))
        if not text_col or not bias_col:
            print(f"  ⚠ Missing text/bias column in {path}, skipping")
            return
        for row in reader:
            raw = (row.get(text_col) or "").strip()
            if not raw:
                continue
            text = clean_tweet_text(raw)
            if len(text) < MIN_LEN:
                continue
            biased = str(row.get(bias_col) or "").strip().lower()
            if biased in ("1", "true", "yes"):
                label = LABEL_POS
            elif biased in ("0", "false", "no"):
                label = LABEL_NEG
            else:
                continue
            yield text, label, source


def _read_conan_jews(path):
    """Yield Jewish-targeted CONAN hate speech (positive) and its
    counter-narrative (negative). Counter-narratives are expert-written
    rebuttals — fluent text that discusses the same topic without being
    hateful, i.e. excellent additional hard negatives."""
    if not path.exists():
        print(f"  ⚠ Not found, skipping: {path}")
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("TARGET") or "").strip().upper() != "JEWS":
                continue
            hs = clean_tweet_text((row.get("HATE_SPEECH") or "").strip())
            cn = clean_tweet_text((row.get("COUNTER_NARRATIVE") or "").strip())
            if len(hs) >= MIN_LEN:
                yield hs, LABEL_POS, "conan_multitarget"
            if len(cn) >= MIN_LEN:
                yield cn, LABEL_NEG, "conan_multitarget"


def build_dataset(
    max_per_class=1500,
    dev_frac=0.15,
    test_frac=0.15,
    include_conan=False,
    seed=RANDOM_SEED,
):
    """Build balanced train/dev/test splits and write them as JSONL.

    Balancing to an equal positive/negative count keeps exact-match accuracy
    meaningful and prevents the optimizer from exploiting class skew.
    """
    rng = random.Random(seed)

    rows = []
    print("Reading ISCA datasets...")
    for path, source in ISCA_FILES:
        before = len(rows)
        rows.extend(_read_isca(path, source))
        print(f"  {source}: +{len(rows) - before}")
    if include_conan:
        print("Reading CONAN (JEWS target)...")
        before = len(rows)
        rows.extend(_read_conan_jews(CONAN_MULTITARGET))
        print(f"  conan_multitarget: +{len(rows) - before}")

    # Deduplicate on cleaned text (keeps first occurrence / its source).
    seen = set()
    deduped = []
    for text, label, source in rows:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((text, label, source))

    pos = [r for r in deduped if r[1] == LABEL_POS]
    neg = [r for r in deduped if r[1] == LABEL_NEG]
    print(f"\nDeduplicated pool: {len(pos)} positive, {len(neg)} negative")

    # Balance: take an equal number per class, capped by max_per_class.
    n = min(len(pos), len(neg), max_per_class)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos, neg = pos[:n], neg[:n]
    print(f"Balanced to {n} per class ({2 * n} total)")

    examples = pos + neg
    rng.shuffle(examples)

    # Split.
    n_total = len(examples)
    n_test = int(n_total * test_frac)
    n_dev = int(n_total * dev_frac)
    test = examples[:n_test]
    dev = examples[n_test:n_test + n_dev]
    train = examples[n_test + n_dev:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in ((train, TRAIN_PATH), (dev, DEV_PATH), (test, TEST_PATH)):
        with open(path, "w", encoding="utf-8") as f:
            for text, label, source in split:
                f.write(json.dumps({"text": text, "label": label, "source": source}) + "\n")
        n_pos = sum(1 for r in split if r[1] == LABEL_POS)
        print(f"  wrote {path.name}: {len(split)} examples ({n_pos} pos / {len(split) - n_pos} neg)")

    return {"train": len(train), "dev": len(dev), "test": len(test)}


def load_split(path):
    """Load a JSONL split as a list of dspy.Example objects.

    Each example exposes ``text`` as the input and ``label`` as the gold
    output. ``with_inputs("text")`` tells DSPy which field is the input.
    """
    import dspy

    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ex = dspy.Example(text=row["text"], label=row["label"]).with_inputs("text")
            examples.append(ex)
    return examples


def load_all():
    """Convenience: load (train, dev, test) as dspy.Example lists."""
    for path in (TRAIN_PATH, DEV_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build the dataset first:\n"
                f"    python -m classifier.data --build"
            )
    return load_split(TRAIN_PATH), load_split(DEV_PATH), load_split(TEST_PATH)


def main():
    p = argparse.ArgumentParser(description="Build the classifier dataset from raw sources.")
    p.add_argument("--build", action="store_true", help="Build train/dev/test JSONL splits")
    p.add_argument("--max-per-class", type=int, default=1500)
    p.add_argument("--include-conan", action="store_true",
                   help="Also include CONAN Jewish-targeted hate speech + counter-narratives")
    args = p.parse_args()

    if args.build:
        build_dataset(max_per_class=args.max_per_class, include_conan=args.include_conan)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
