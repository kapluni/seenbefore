"""
config.py — Central configuration for the DSPy antisemitism classifier
======================================================================
Model names, paths, and label constants live here so the rest of the
package stays free of magic strings.

Models are resolved through litellm (DSPy's backend). Override any of them
with environment variables without touching code:

    CLASSIFIER_TASK_MODEL    — the model that does the actual classification
                               (default: fast/cheap Haiku)
    CLASSIFIER_PROMPT_MODEL  — the model that *proposes* better prompts during
                               optimization (default: stronger Sonnet)

An ANTHROPIC_API_KEY must be set for tuning, evaluation, and prediction.
Building the dataset needs no API key.
"""

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DATA_DIR = PKG_DIR / "data"
ARTIFACTS_DIR = PKG_DIR / "artifacts"   # optimized programs are saved here

RAW_SOURCES_DIR = REPO_ROOT / "corpus" / "modern_sources"

TRAIN_PATH = DATA_DIR / "train.jsonl"
DEV_PATH = DATA_DIR / "dev.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

OPTIMIZED_PROGRAM_PATH = ARTIFACTS_DIR / "optimized_classifier.json"

# --- Labels ----------------------------------------------------------------
# The classifier emits one of these two string labels. Keeping them as
# explicit constants avoids typos leaking into the metric and the prompt.
LABEL_POS = "ANTISEMITIC"
LABEL_NEG = "NOT_ANTISEMITIC"
LABELS = (LABEL_POS, LABEL_NEG)

# --- Models ----------------------------------------------------------------
# Defaults favour a fast, inexpensive model for the high-volume classification
# task and a stronger model for the (much rarer) prompt-proposal step during
# optimization. Both are overridable via environment variables.
TASK_MODEL = os.environ.get("CLASSIFIER_TASK_MODEL", "anthropic/claude-haiku-4-5-20251001")
PROMPT_MODEL = os.environ.get("CLASSIFIER_PROMPT_MODEL", "anthropic/claude-sonnet-4-6")

# Generation settings for the task model. Temperature 0 keeps classification
# deterministic and reproducible.
TASK_TEMPERATURE = float(os.environ.get("CLASSIFIER_TASK_TEMPERATURE", "0.0"))
TASK_MAX_TOKENS = int(os.environ.get("CLASSIFIER_TASK_MAX_TOKENS", "1024"))

# Reproducibility for dataset splitting / sampling.
RANDOM_SEED = int(os.environ.get("CLASSIFIER_SEED", "42"))
