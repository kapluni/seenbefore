"""
classifier — LLM-based antisemitism classifier with DSPy auto prompt tuning
===========================================================================
A self-contained sub-project of "I've Seen This Before". Where the parent
project maps Soviet propaganda to modern rhetoric via embeddings, this package
builds a *supervised, prompt-optimized* LLM classifier that labels individual
social-media posts as ANTISEMITIC / NOT_ANTISEMITIC.

Pipeline:
    data.py     → build balanced train/dev/test from the labeled corpus
    signature.py→ the initial, hand-written prompt (IHRA + 3D + 9 tropes)
    program.py  → the DSPy chain-of-thought classifier module
    metrics.py  → exact-match + credibility-weighted metrics, full reports
    tune.py     → DSPy MIPROv2 / Bootstrap auto prompt tuning
    evaluate.py → baseline vs. optimized on the held-out test set
    predict.py  → inference (CLI + importable)

See classifier/README.md for the quick start.
"""

from .config import LABEL_POS, LABEL_NEG, LABELS  # noqa: F401
