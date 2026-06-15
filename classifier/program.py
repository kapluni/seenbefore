"""
program.py — The classifier as a DSPy module
============================================
Wraps the ``ClassifyAntisemitism`` signature in a ``dspy.Module``. We use
``dspy.ChainOfThought`` so the model reasons before committing to a label —
this materially improves the legitimate-criticism vs. antisemitism boundary,
which is the hard part of the task.

The module is deliberately thin: all the "intelligence" lives in the
signature's instructions and in the few-shot demonstrations that the
optimizer attaches during tuning. That separation is what makes auto prompt
tuning possible — the optimizer rewrites the predictor inside this module,
not the module's Python.
"""

import dspy

from .signature import ClassifyAntisemitism
from .config import LABEL_POS, LABEL_NEG


def normalize_label(raw):
    """Coerce a free-text model output into one of the two canonical labels.

    The LM is instructed to emit an exact label, but tuning checkpoints and
    temperature can produce minor variants ("antisemitic.", "not antisemitic").
    Normalizing here keeps the metric honest without masking real errors.
    """
    if not raw:
        return LABEL_NEG
    t = str(raw).strip().upper()
    # Order matters: check the negative form first because it contains the
    # positive label as a substring ("NOT_ANTISEMITIC" contains "ANTISEMITIC").
    if "NOT" in t or t.startswith("NEG") or "NOT_ANTISEMITIC" in t:
        return LABEL_NEG
    if "ANTISEMIT" in t or t.startswith("POS"):
        return LABEL_POS
    return LABEL_NEG


class AntisemitismClassifier(dspy.Module):
    """Chain-of-thought classifier returning a normalized label + rationale."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifyAntisemitism)

    def forward(self, text):
        pred = self.classify(text=text)
        label = normalize_label(getattr(pred, "label", ""))
        return dspy.Prediction(
            label=label,
            rationale=getattr(pred, "rationale", ""),
        )
