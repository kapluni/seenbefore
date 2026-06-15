# Antisemitism Classifier (DSPy auto-prompt-tuning)

An LLM-based classifier that labels individual social-media posts as
**ANTISEMITIC** or **NOT_ANTISEMITIC**, with automatic prompt optimization via
[DSPy](https://dspy.ai).

This is a sub-project of **"I've Seen This Before."** The parent project maps
Soviet propaganda to modern rhetoric using embeddings; this one builds a
*supervised, prompt-optimized* classifier on top of the same labeled corpus.

## Why DSPy

Rather than hand-tuning a prompt by trial and error, we write a clear initial
prompt (the IHRA working definition + the "3D test" + the project's 9-trope
taxonomy) and let DSPy's optimizers automatically:

1. **rewrite the instructions** (proposed by a stronger model), and
2. **bootstrap few-shot demonstrations** from the labeled training data,

searching for the combination that maximizes accuracy on a held-out dev set.
The result is a reproducible, measurable prompt — not a guess.

## The core design constraint

The hardest and most important judgment is **antisemitism vs. legitimate
criticism of Israel**. Every negative example in our data already mentions
Jews / Israel / Zionism (the ISCA datasets surfaced them by those keywords, then
humans labeled them "not biased"). So the classifier is trained and scored
against *hard negatives* — exactly the boundary the project's credibility
depends on. The `safety` metric and the reported **false-positive rate** exist
to keep wrongful flagging of legitimate criticism low.

## Layout

| File | Role |
|------|------|
| `config.py` | Models, paths, label constants (override models via env vars) |
| `data.py` | ETL: raw datasets → balanced `train/dev/test.jsonl`; loaders |
| `signature.py` | The **initial prompt** (IHRA + 3D test + 9 tropes) as a DSPy Signature |
| `program.py` | The chain-of-thought classifier `dspy.Module` |
| `metrics.py` | Exact-match + credibility-weighted metrics; full report (P/R/F1, FP-rate) |
| `lm.py` | Configures the task model and the prompt-proposal model |
| `tune.py` | **Auto prompt tuning** (MIPROv2 / Bootstrap); saves the optimized program |
| `evaluate.py` | Baseline vs. optimized on the held-out test set |
| `predict.py` | Inference — CLI and importable |

## Quick start

```bash
# 1. Install deps (parent venv + these)
pip install -r classifier/requirements.txt

# 2. Build the dataset (no API key needed)
python -m classifier.data --build
#   → balanced train/dev/test JSONL in classifier/data/
#   add --include-conan to also pull CONAN Jewish-targeted hate speech + counter-narratives

# 3. Auto-tune the prompt (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python -m classifier.tune                      # MIPROv2, light budget
#   variants:
python -m classifier.tune --auto medium        # larger search budget
python -m classifier.tune --metric safety       # optimize the credibility-weighted metric
python -m classifier.tune --optimizer bootstrap # cheaper: demos only, keep instructions

# 4. Evaluate baseline vs. tuned on the held-out test set
python -m classifier.evaluate

# 5. Classify new text
python -m classifier.predict "Zionists run the banks and the media."
echo "Criticizing settlement expansion is fair policy debate." | python -m classifier.predict --stdin
python -m classifier.predict --file posts.txt --json
```

### As a library

```python
from classifier.predict import load_classifier

clf = load_classifier()                 # uses the tuned program if available
r = clf("Israel has no right to exist — the Zionist entity must be dismantled.")
print(r["label"], "—", r["rationale"])
```

## Models

Defaults (override with environment variables):

- `CLASSIFIER_TASK_MODEL` — `anthropic/claude-haiku-4-5-20251001` (fast, does the classifying)
- `CLASSIFIER_PROMPT_MODEL` — `anthropic/claude-sonnet-4-6` (proposes better prompts during tuning)

DSPy reaches Anthropic through litellm, hence the `anthropic/` prefix.

## Dataset

Built from the parent project's downloaded sources (`corpus/modern_sources/`):

- **ISCA** (Zenodo, ClassData 2022–2023, HuggingFace) — IHRA-labeled tweets via
  the `Biased` column. `Biased=1` → positive, `Biased=0` → hard negative.
- **CONAN** (optional, `--include-conan`) — Jewish-targeted expert hate speech
  (positive) + its counter-narratives (negative).

Examples are cleaned with the parent project's `clean_tweet_text`, deduplicated,
and balanced to an equal positive/negative count so accuracy stays meaningful.
With the default cap this yields 1,500 per class (3,000 total), split 70/15/15.

## Metrics

- **accuracy** (default) — exact-match; simple and interpretable.
- **safety** — credibility-weighted: a false positive (flagging legitimate
  criticism) scores 0; a false negative earns partial credit. Nudges the
  optimizer toward precision on the positive class.

Every evaluation also reports precision, recall, F1, the confusion matrix, and
the **false-positive rate** — the share of legitimate posts wrongly flagged.

## Notes & next steps

- Tuning, evaluation, and prediction require `ANTHROPIC_API_KEY`. Building the
  dataset does not.
- The optimized program is saved to `classifier/artifacts/` as JSON (tuned
  instructions + selected demonstrations) and reloaded by `evaluate.py` /
  `predict.py` — no need to re-run the (paid) optimization.
- Possible extensions: emit the predicted trope category as a structured field;
  add a `GEPA` reflective-evolution optimizer; cross-validate the
  legitimate-criticism boundary against the parent project's 8 calibration
  texts; export predictions back into `viz_data.json`.
```
