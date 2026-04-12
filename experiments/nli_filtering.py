"""
NLI Filtering Experiment
========================
Tests whether Natural Language Inference (entailment/contradiction/neutral)
can distinguish genuine rhetorical echoes from "same topic, opposite argument"
false positives — the core precision problem in the pipeline.

Uses cross-encoder/nli-deberta-v3-base which outputs [contradiction, entailment, neutral]
probabilities for each text pair.

Run: python experiments/nli_filtering.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import CrossEncoder
import numpy as np
from generate_viz_data import SOVIET_SEED_PASSAGES, LEGITIMATE_TEXTS, SAMPLE_MODERN
from embedding_pipeline import EmbeddingEngine, CONFIG

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
STS_MODEL = "cross-encoder/stsb-roberta-large"

# Labels for NLI output indices
NLI_LABELS = ["contradiction", "entailment", "neutral"]

def load_viz_matches():
    """Load curated matches from viz_data.json."""
    with open("viz_data.json") as f:
        data = json.load(f)
    return data["matches"], data.get("calibration", {})

def find_best_soviet_match(text, soviet_passages, engine):
    """Find the most similar Soviet passage for a given text."""
    query_emb = engine.embed_query(text)
    soviet_embs = np.array([p["embedding"] for p in soviet_passages])
    sims = np.dot(soviet_embs, query_emb)
    best_idx = int(np.argmax(sims))
    return soviet_passages[best_idx], float(sims[best_idx])

def embed_soviet_seeds(engine):
    """Embed the Soviet seed passages for matching."""
    texts = [p["text"] for p in SOVIET_SEED_PASSAGES]
    embeddings = engine.model.encode(texts, normalize_embeddings=True)
    result = []
    for i, p in enumerate(SOVIET_SEED_PASSAGES):
        item = dict(p)
        item["embedding"] = embeddings[i]
        result.append(item)
    return result

def run_nli_on_pairs(nli_model, pairs, labels):
    """Run NLI model on a list of (text_a, text_b) pairs.
    Returns list of dicts with scores for each label."""
    raw_scores = nli_model.predict(pairs)
    results = []
    for scores in raw_scores:
        # Apply softmax if raw logits
        if hasattr(scores, '__len__'):
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            results.append({
                NLI_LABELS[i]: float(probs[i]) for i in range(3)
            })
        else:
            results.append({"score": float(scores)})
    return results

def run_sts_on_pairs(sts_model, pairs):
    """Run STS cross-encoder on pairs. Returns list of float scores."""
    scores = sts_model.predict(pairs)
    return [float(s) for s in scores]

def format_nli(nli_result):
    """Format NLI result as a compact string."""
    e = nli_result["entailment"]
    c = nli_result["contradiction"]
    n = nli_result["neutral"]
    winner = max(NLI_LABELS, key=lambda l: nli_result[l])
    return f"E={e:.3f} C={c:.3f} N={n:.3f} [{winner.upper()}]"

def main():
    print("=" * 70)
    print("NLI FILTERING EXPERIMENT")
    print("=" * 70)

    # Load models
    print(f"\nLoading NLI model: {NLI_MODEL}")
    nli_model = CrossEncoder(NLI_MODEL)
    print(f"Loading STS model: {STS_MODEL}")
    sts_model = CrossEncoder(STS_MODEL)

    # Load embedding model for matching calibration/sample texts to Soviet passages
    print(f"Loading embedding model: {CONFIG['default_model']}")
    engine = EmbeddingEngine(CONFIG["default_model"])
    engine.load_model()
    soviet_embedded = embed_soviet_seeds(engine)

    # -------------------------------------------------------------------
    # 1. Run on existing curated matches from viz_data.json
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 1: Curated Matches (from viz_data.json)")
    print("=" * 70)

    matches, calibration = load_viz_matches()
    match_pairs = [
        (m.get("sovietTextFull") or m["sovietText"],
         m.get("modernTextFull") or m["modernText"])
        for m in matches
    ]

    nli_results = run_nli_on_pairs(nli_model, match_pairs, NLI_LABELS)
    sts_results = run_sts_on_pairs(sts_model, match_pairs)

    print(f"\n{'#':>2} {'Cosine':>7} {'STS':>7} {'NLI Entail':>11} {'NLI Contr':>10} {'NLI Neut':>9} {'Winner':>14}")
    print("-" * 70)
    for i, (m, nli, sts) in enumerate(zip(matches, nli_results, sts_results)):
        winner = max(NLI_LABELS, key=lambda l: nli[l])
        print(f"{i+1:>2} {m['similarity']:>7.4f} {sts:>7.4f} {nli['entailment']:>11.4f} {nli['contradiction']:>10.4f} {nli['neutral']:>9.4f} {winner.upper():>14}")

    # -------------------------------------------------------------------
    # 2. Run on SAMPLE_MODERN (known strong echoes) vs best Soviet match
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 2: Known Strong Echoes (SAMPLE_MODERN — hand-crafted)")
    print("=" * 70)

    echo_pairs = []
    echo_info = []
    for mod in SAMPLE_MODERN:
        best_sov, sim = find_best_soviet_match(mod["text"], soviet_embedded, engine)
        echo_pairs.append((best_sov["text"], mod["text"]))
        echo_info.append({"modern": mod["text"][:80], "soviet": best_sov["text"][:80], "cosine": sim})

    echo_nli = run_nli_on_pairs(nli_model, echo_pairs, NLI_LABELS)
    echo_sts = run_sts_on_pairs(sts_model, echo_pairs)

    print(f"\n{'#':>2} {'Cosine':>7} {'STS':>7} {'Entail':>7} {'Contr':>7} {'Neut':>7} {'Winner':>14} Modern text")
    print("-" * 100)
    for i, (info, nli, sts) in enumerate(zip(echo_info, echo_nli, echo_sts)):
        winner = max(NLI_LABELS, key=lambda l: nli[l])
        print(f"{i+1:>2} {info['cosine']:>7.4f} {sts:>7.4f} {nli['entailment']:>7.4f} {nli['contradiction']:>7.4f} {nli['neutral']:>7.4f} {winner.upper():>14} {info['modern'][:50]}...")

    # -------------------------------------------------------------------
    # 3. Run on LEGITIMATE_TEXTS (should NOT be entailment)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 3: Legitimate Criticism (should be contradiction/neutral)")
    print("=" * 70)

    legit_pairs = []
    legit_info = []
    for item in LEGITIMATE_TEXTS:
        best_sov, sim = find_best_soviet_match(item["text"], soviet_embedded, engine)
        legit_pairs.append((best_sov["text"], item["text"]))
        legit_info.append({"modern": item["text"][:80], "source": item["source"], "cosine": sim})

    legit_nli = run_nli_on_pairs(nli_model, legit_pairs, NLI_LABELS)
    legit_sts = run_sts_on_pairs(sts_model, legit_pairs)

    print(f"\n{'#':>2} {'Cosine':>7} {'STS':>7} {'Entail':>7} {'Contr':>7} {'Neut':>7} {'Winner':>14} Source")
    print("-" * 90)
    for i, (info, nli, sts) in enumerate(zip(legit_info, legit_nli, legit_sts)):
        winner = max(NLI_LABELS, key=lambda l: nli[l])
        print(f"{i+1:>2} {info['cosine']:>7.4f} {sts:>7.4f} {nli['entailment']:>7.4f} {nli['contradiction']:>7.4f} {nli['neutral']:>7.4f} {winner.upper():>14} {info['source']}")

    # -------------------------------------------------------------------
    # 4. Adversarial pairs: same topic, opposite argument
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 4: Adversarial Pairs (same topic, opposite argument)")
    print("=" * 70)
    print("These should score HIGH on cosine (same topic) but HIGH on contradiction (opposite stance)")

    adversarial_modern = [
        {"text": "Zionism is the national liberation movement of the Jewish people, comparable to other national movements.", "label": "Pro-Zionism (nationalism)"},
        {"text": "Israel is a thriving democracy that grants equal rights to all citizens regardless of ethnicity.", "label": "Pro-Israel (democracy)"},
        {"text": "Accusations of Zionist media control are classic antisemitic conspiracy theories with no basis in fact.", "label": "Anti-conspiracy"},
        {"text": "Israel's founding was a legitimate act of self-determination by a persecuted people returning to their ancestral homeland.", "label": "Pro-Zionism (self-determination)"},
        {"text": "The charge that Zionism equals racism was a Soviet propaganda invention that was rightly revoked by the UN in 1991.", "label": "Anti-Zionism=racism"},
    ]

    # Pair each adversarial text with the most thematically relevant Soviet passage
    adv_pairs = []
    adv_info = []
    for item in adversarial_modern:
        best_sov, sim = find_best_soviet_match(item["text"], soviet_embedded, engine)
        adv_pairs.append((best_sov["text"], item["text"]))
        adv_info.append({"label": item["label"], "cosine": sim, "soviet": best_sov["text"][:80]})

    adv_nli = run_nli_on_pairs(nli_model, adv_pairs, NLI_LABELS)
    adv_sts = run_sts_on_pairs(sts_model, adv_pairs)

    print(f"\n{'#':>2} {'Cosine':>7} {'STS':>7} {'Entail':>7} {'Contr':>7} {'Neut':>7} {'Winner':>14} Label")
    print("-" * 90)
    for i, (info, nli, sts) in enumerate(zip(adv_info, adv_nli, adv_sts)):
        winner = max(NLI_LABELS, key=lambda l: nli[l])
        print(f"{i+1:>2} {info['cosine']:>7.4f} {sts:>7.4f} {nli['entailment']:>7.4f} {nli['contradiction']:>7.4f} {nli['neutral']:>7.4f} {winner.upper():>14} {info['label']}")

    # -------------------------------------------------------------------
    # 5. Summary and threshold analysis
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 5: Summary & Threshold Analysis")
    print("=" * 70)

    all_echo_entail = [r["entailment"] for r in echo_nli]
    all_match_entail = [r["entailment"] for r in nli_results]
    all_legit_entail = [r["entailment"] for r in legit_nli]
    all_adv_entail = [r["entailment"] for r in adv_nli]

    print(f"\nEntailment score distributions:")
    print(f"  Known echoes (SAMPLE_MODERN):  mean={np.mean(all_echo_entail):.4f}  min={np.min(all_echo_entail):.4f}  max={np.max(all_echo_entail):.4f}")
    print(f"  Curated matches (viz_data):    mean={np.mean(all_match_entail):.4f}  min={np.min(all_match_entail):.4f}  max={np.max(all_match_entail):.4f}")
    print(f"  Legitimate criticism:          mean={np.mean(all_legit_entail):.4f}  min={np.min(all_legit_entail):.4f}  max={np.max(all_legit_entail):.4f}")
    print(f"  Adversarial (opposite stance): mean={np.mean(all_adv_entail):.4f}  min={np.min(all_adv_entail):.4f}  max={np.max(all_adv_entail):.4f}")

    all_echo_contr = [r["contradiction"] for r in echo_nli]
    all_match_contr = [r["contradiction"] for r in nli_results]
    all_legit_contr = [r["contradiction"] for r in legit_nli]
    all_adv_contr = [r["contradiction"] for r in adv_nli]

    print(f"\nContradiction score distributions:")
    print(f"  Known echoes (SAMPLE_MODERN):  mean={np.mean(all_echo_contr):.4f}  min={np.min(all_echo_contr):.4f}  max={np.max(all_echo_contr):.4f}")
    print(f"  Curated matches (viz_data):    mean={np.mean(all_match_contr):.4f}  min={np.min(all_match_contr):.4f}  max={np.max(all_match_contr):.4f}")
    print(f"  Legitimate criticism:          mean={np.mean(all_legit_contr):.4f}  min={np.min(all_legit_contr):.4f}  max={np.max(all_legit_contr):.4f}")
    print(f"  Adversarial (opposite stance): mean={np.mean(all_adv_contr):.4f}  min={np.min(all_adv_contr):.4f}  max={np.max(all_adv_contr):.4f}")

    # Threshold analysis: at various entailment thresholds, how many from each group pass?
    print(f"\nThreshold analysis — matches passing at entailment >= T:")
    print(f"{'Threshold':>10} {'Echoes':>8} {'Matches':>9} {'Legit':>7} {'Advers':>8}")
    print("-" * 45)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        e_pass = sum(1 for x in all_echo_entail if x >= t)
        m_pass = sum(1 for x in all_match_entail if x >= t)
        l_pass = sum(1 for x in all_legit_entail if x >= t)
        a_pass = sum(1 for x in all_adv_entail if x >= t)
        print(f"{t:>10.1f} {e_pass:>5}/{len(all_echo_entail)} {m_pass:>6}/{len(all_match_entail)} {l_pass:>4}/{len(all_legit_entail)} {a_pass:>5}/{len(all_adv_entail)}")

    # Contradiction filter: at various thresholds, how many from each group get filtered OUT?
    print(f"\nContradiction filter — matches REMOVED at contradiction >= T:")
    print(f"{'Threshold':>10} {'Echoes':>8} {'Matches':>9} {'Legit':>7} {'Advers':>8}")
    print("-" * 45)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        e_fail = sum(1 for x in all_echo_contr if x >= t)
        m_fail = sum(1 for x in all_match_contr if x >= t)
        l_fail = sum(1 for x in all_legit_contr if x >= t)
        a_fail = sum(1 for x in all_adv_contr if x >= t)
        print(f"{t:>10.1f} {e_fail:>5}/{len(all_echo_contr)} {m_fail:>6}/{len(all_match_contr)} {l_fail:>4}/{len(all_legit_contr)} {a_fail:>5}/{len(all_adv_contr)}")

    # -------------------------------------------------------------------
    # 6. Write results to markdown
    # -------------------------------------------------------------------
    write_results_md(matches, nli_results, sts_results,
                     echo_info, echo_nli, echo_sts,
                     legit_info, legit_nli, legit_sts,
                     adv_info, adv_nli, adv_sts)

    print(f"\nResults written to experiments/nli_results.md")
    print("Done!")


def write_results_md(matches, match_nli, match_sts,
                     echo_info, echo_nli, echo_sts,
                     legit_info, legit_nli, legit_sts,
                     adv_info, adv_nli, adv_sts):
    """Write detailed results to markdown file."""
    lines = []
    lines.append("# NLI Filtering Experiment Results\n")
    lines.append(f"Model: `{NLI_MODEL}`\n")
    lines.append("NLI outputs 3 probabilities: **entailment** (same claim), **contradiction** (opposite claim), **neutral** (unrelated).\n")
    lines.append("The hypothesis: genuine rhetorical echoes should score high on entailment, while same-topic-opposite-argument false positives should score high on contradiction.\n")

    # Section 1: Curated matches
    lines.append("## 1. Curated Matches (viz_data.json)\n")
    lines.append("| # | Cosine | STS | Entail | Contr | Neutral | Winner | Soviet (first 60) | Modern (first 60) |")
    lines.append("|---|--------|-----|--------|-------|---------|--------|-------------------|-------------------|")
    for i, (m, nli, sts) in enumerate(zip(matches, match_nli, match_sts)):
        winner = max(NLI_LABELS, key=lambda l: nli[l]).upper()
        sov = (m.get("sovietTextFull") or m["sovietText"])[:60].replace("|", "/")
        mod = (m.get("modernTextFull") or m["modernText"])[:60].replace("|", "/")
        lines.append(f"| {i+1} | {m['similarity']:.4f} | {sts:.4f} | {nli['entailment']:.4f} | {nli['contradiction']:.4f} | {nli['neutral']:.4f} | {winner} | {sov}... | {mod}... |")

    # Section 2: Known echoes
    lines.append("\n## 2. Known Strong Echoes (SAMPLE_MODERN)\n")
    lines.append("These are hand-crafted texts that are *known* to echo Soviet propaganda. Entailment should be high.\n")
    lines.append("| # | Cosine | STS | Entail | Contr | Neutral | Winner | Modern text |")
    lines.append("|---|--------|-----|--------|-------|---------|--------|-------------|")
    for i, (info, nli, sts) in enumerate(zip(echo_info, echo_nli, echo_sts)):
        winner = max(NLI_LABELS, key=lambda l: nli[l]).upper()
        lines.append(f"| {i+1} | {info['cosine']:.4f} | {sts:.4f} | {nli['entailment']:.4f} | {nli['contradiction']:.4f} | {nli['neutral']:.4f} | {winner} | {info['modern'][:70]}... |")

    # Section 3: Legitimate criticism
    lines.append("\n## 3. Legitimate Criticism (should NOT be entailment)\n")
    lines.append("| # | Cosine | STS | Entail | Contr | Neutral | Winner | Source |")
    lines.append("|---|--------|-----|--------|-------|---------|--------|--------|")
    for i, (info, nli, sts) in enumerate(zip(legit_info, legit_nli, legit_sts)):
        winner = max(NLI_LABELS, key=lambda l: nli[l]).upper()
        lines.append(f"| {i+1} | {info['cosine']:.4f} | {sts:.4f} | {nli['entailment']:.4f} | {nli['contradiction']:.4f} | {nli['neutral']:.4f} | {winner} | {info['source']} |")

    # Section 4: Adversarial pairs
    lines.append("\n## 4. Adversarial Pairs (same topic, opposite stance)\n")
    lines.append("These discuss the same topics as Soviet propaganda but take the opposite position. Contradiction should be high.\n")
    lines.append("| # | Cosine | STS | Entail | Contr | Neutral | Winner | Label |")
    lines.append("|---|--------|-----|--------|-------|---------|--------|-------|")
    for i, (info, nli, sts) in enumerate(zip(adv_info, adv_nli, adv_sts)):
        winner = max(NLI_LABELS, key=lambda l: nli[l]).upper()
        lines.append(f"| {i+1} | {info['cosine']:.4f} | {sts:.4f} | {nli['entailment']:.4f} | {nli['contradiction']:.4f} | {nli['neutral']:.4f} | {winner} | {info['label']} |")

    # Section 5: Summary
    lines.append("\n## 5. Summary Statistics\n")

    all_groups = [
        ("Known echoes", echo_nli),
        ("Curated matches", match_nli),
        ("Legitimate criticism", legit_nli),
        ("Adversarial", adv_nli),
    ]
    lines.append("| Group | Mean Entail | Mean Contr | Mean Neutral |")
    lines.append("|-------|-------------|------------|--------------|")
    for name, nli_list in all_groups:
        me = np.mean([r["entailment"] for r in nli_list])
        mc = np.mean([r["contradiction"] for r in nli_list])
        mn = np.mean([r["neutral"] for r in nli_list])
        lines.append(f"| {name} | {me:.4f} | {mc:.4f} | {mn:.4f} |")

    lines.append("\n## 6. Recommendation\n")
    lines.append("*To be filled in after reviewing results.*\n")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/nli_results.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
