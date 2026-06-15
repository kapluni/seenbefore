"""
lm.py — Configure the DSPy language models
==========================================
Centralizes LM setup so tuning, evaluation, and prediction all use the same
configuration. DSPy talks to Anthropic through litellm, so model names use the
``anthropic/<model-id>`` form (see config.py).

Requires ANTHROPIC_API_KEY in the environment.
"""

import os
import dspy

from .config import TASK_MODEL, PROMPT_MODEL, TASK_TEMPERATURE, TASK_MAX_TOKENS


def _require_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before tuning, evaluating, "
            "or predicting:\n    export ANTHROPIC_API_KEY=sk-ant-..."
        )


def configure_task_lm():
    """Set the global DSPy LM to the (fast) classification model and return it."""
    _require_key()
    lm = dspy.LM(TASK_MODEL, temperature=TASK_TEMPERATURE, max_tokens=TASK_MAX_TOKENS)
    dspy.configure(lm=lm)
    return lm


def make_prompt_lm():
    """Return the stronger model used to *propose* prompts during optimization.

    A small bit of temperature helps the optimizer explore diverse instruction
    candidates rather than collapsing to one phrasing.
    """
    _require_key()
    return dspy.LM(PROMPT_MODEL, temperature=0.7, max_tokens=4096)
