#!/usr/bin/env python3
"""
Phase 2 benchmark: deeper investigation of the most promising approaches.

Findings from Phase 1:
- cross-encoder/stsb-roberta-large: BEST separation (0.2249) but it's a cross-encoder
- All bi-encoders have POOR separation (<0.1)
- The TRICKY pairs (opposite arguments) are the hardest - all bi-encoders score them HIGH
- NLI models (deberta) saturate at ~1.0 for everything in this domain

Phase 2 tests:
1. Two-stage pipeline: bi-encoder retrieval + cross-encoder reranking
2. NLI entailment vs contradiction breakdown (use full 3-class output)
3. all-mpnet-base-v2 (popular general-purpose)
4. BGE-reranker-v2-m3 (purpose-built reranker)
5. Hybrid score: cosine_sim * entailment_prob
"""

import time
import numpy as np
import torch
import gc

# ============================================================
# TEST PAIRS (same as phase 1)
# ============================================================

GOOD_PAIRS = [
    ("Zionism concentrates ultra-nationalism, chauvinism and racial intolerance",
     "Zionism is a racist, settler colonialist movement practicing apartheid"),
    ("International Zionist Organization controls significant part of media agencies",
     "The Jews control everything, the media, education, and the economy"),
    ("The Hitlerites acted in the same way, Zionism-Fascism-Hitlerism",
     "Zionists are basically Nazis and not really Jews at all"),
]

BAD_PAIRS = [
    ("Zionism concentrates ultra-nationalism and racial intolerance",
     "I disagree with Israeli settlement policy in the West Bank"),
    ("International Zionist Organization controls media",
     "Israeli tech startups have been successful in media technology"),
    ("The Hitlerites acted in the same way",
     "The Holocaust was a tragedy that must never be forgotten"),
]

TRICKY_PAIRS = [
    ("Absurd are attempts to present criticism of Zionism as antisemitic",
     "Criticism of Israel often crosses the line into antisemitism and should be called out"),
    ("Zionism is the front squad of colonialism",
     "Israel was founded as a refuge for persecuted Jews, not as a colonial project"),
]

ALL_PAIRS = [("GOOD", a, b) for a, b in GOOD_PAIRS] + \
            [("BAD", a, b) for a, b in BAD_PAIRS] + \
            [("TRICKY", a, b) for a, b in TRICKY_PAIRS]


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# TEST 1: Full NLI breakdown (3 classes)
# ============================================================

def test_nli_full_breakdown():
    """Show contradiction/neutral/entailment for each pair."""
    from sentence_transformers import CrossEncoder

    print("\n" + "="*70)
    print("NLI FULL BREAKDOWN: cross-encoder/nli-deberta-v3-large")
    print("="*70)

    model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
    pairs = [(a, b) for _, a, b in ALL_PAIRS]
    raw = model.predict(pairs)
    raw = np.array(raw)

    print(f"{'Cat':>7} {'Contra':>8} {'Neutral':>8} {'Entail':>8}  Texts")
    print("-" * 80)

    for i, (cat, a, b) in enumerate(ALL_PAIRS):
        probs = np.exp(raw[i]) / np.exp(raw[i]).sum()
        contra, neutral, entail = probs
        print(f"[{cat:>6}] {contra:>8.4f} {neutral:>8.4f} {entail:>8.4f}")
        print(f"         S: {a[:60]}")
        print(f"         M: {b[:60]}")

    del model
    cleanup()


# ============================================================
# TEST 2: STSB cross-encoder detailed analysis
# ============================================================

def test_stsb_detailed():
    """Detailed STSB cross-encoder analysis - our best performer."""
    from sentence_transformers import CrossEncoder

    print("\n" + "="*70)
    print("DETAILED: cross-encoder/stsb-roberta-large")
    print("="*70)

    model = CrossEncoder("cross-encoder/stsb-roberta-large")
    pairs = [(a, b) for _, a, b in ALL_PAIRS]
    scores = model.predict(pairs)

    cat_scores = {"GOOD": [], "BAD": [], "TRICKY": []}
    for i, (cat, a, b) in enumerate(ALL_PAIRS):
        s = float(scores[i])
        cat_scores[cat].append(s)
        print(f"  [{cat:6s}] {s:.4f}  S: {a[:55]}")
        print(f"                    M: {b[:55]}")

    print(f"\n  GOOD range:   [{min(cat_scores['GOOD']):.4f}, {max(cat_scores['GOOD']):.4f}]  avg={np.mean(cat_scores['GOOD']):.4f}")
    print(f"  BAD range:    [{min(cat_scores['BAD']):.4f}, {max(cat_scores['BAD']):.4f}]  avg={np.mean(cat_scores['BAD']):.4f}")
    print(f"  TRICKY range: [{min(cat_scores['TRICKY']):.4f}, {max(cat_scores['TRICKY']):.4f}]  avg={np.mean(cat_scores['TRICKY']):.4f}")

    # Check if threshold exists
    min_good = min(cat_scores['GOOD'])
    max_other = max(max(cat_scores['BAD']), max(cat_scores['TRICKY']))
    if min_good > max_other:
        print(f"  CLEAN THRESHOLD at ~{(min_good + max_other) / 2:.4f}")
    else:
        print(f"  OVERLAP: min(GOOD)={min_good:.4f} vs max(BAD/TRICKY)={max_other:.4f}")

    del model
    cleanup()
    return scores


# ============================================================
# TEST 3: Two-stage pipeline simulation
# ============================================================

def test_two_stage_pipeline():
    """Simulate: BGE retrieval -> STSB reranking."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("\n" + "="*70)
    print("TWO-STAGE PIPELINE: BGE-large retrieval + STSB-roberta reranking")
    print("="*70)

    # Stage 1: BGE bi-encoder scores
    print("  Loading BGE-large-en-v1.5...")
    bi_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    bi_scores = []
    for cat, a, b in ALL_PAIRS:
        ea = bi_model.encode("Represent this sentence: " + a, normalize_embeddings=True)
        eb = bi_model.encode(b, normalize_embeddings=True)
        bi_scores.append(cosine_sim(ea, eb))

    del bi_model
    cleanup()

    # Stage 2: STSB cross-encoder reranking
    print("  Loading STSB cross-encoder...")
    ce_model = CrossEncoder("cross-encoder/stsb-roberta-large")
    pairs = [(a, b) for _, a, b in ALL_PAIRS]
    ce_scores = ce_model.predict(pairs)

    del ce_model
    cleanup()

    # Combined scoring strategies
    print(f"\n{'Cat':>7} {'BiEnc':>7} {'CrossEnc':>9} {'Product':>8} {'Weighted':>9}")
    print("-" * 55)

    cat_combined = {"GOOD": [], "BAD": [], "TRICKY": []}
    for i, (cat, a, b) in enumerate(ALL_PAIRS):
        bi = bi_scores[i]
        ce = float(ce_scores[i])
        product = bi * ce
        weighted = 0.4 * bi + 0.6 * ce  # Weight cross-encoder more
        cat_combined[cat].append(weighted)
        print(f"[{cat:>6}] {bi:>7.4f} {ce:>9.4f} {product:>8.4f} {weighted:>9.4f}")
        print(f"         S: {a[:55]}")
        print(f"         M: {b[:55]}")

    good_avg = np.mean(cat_combined["GOOD"])
    bad_avg = np.mean(cat_combined["BAD"])
    tricky_avg = np.mean(cat_combined["TRICKY"])
    sep = good_avg - max(bad_avg, tricky_avg)
    print(f"\n  Weighted (0.4*bi + 0.6*ce) averages:")
    print(f"    GOOD={good_avg:.4f}  BAD={bad_avg:.4f}  TRICKY={tricky_avg:.4f}")
    print(f"    SEPARATION: {sep:.4f}")


# ============================================================
# TEST 4: all-mpnet-base-v2 (popular general-purpose)
# ============================================================

def test_all_mpnet():
    from sentence_transformers import SentenceTransformer

    print("\n" + "="*70)
    print("TEST: all-mpnet-base-v2")
    print("="*70)

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    cat_scores = {"GOOD": [], "BAD": [], "TRICKY": []}
    for cat, a, b in ALL_PAIRS:
        ea = model.encode(a, normalize_embeddings=True)
        eb = model.encode(b, normalize_embeddings=True)
        s = cosine_sim(ea, eb)
        cat_scores[cat].append(s)
        print(f"  [{cat:6s}] {s:.4f}  {a[:50]}... / {b[:50]}...")

    good_avg = np.mean(cat_scores["GOOD"])
    bad_avg = np.mean(cat_scores["BAD"])
    tricky_avg = np.mean(cat_scores["TRICKY"])
    sep = good_avg - max(bad_avg, tricky_avg)
    print(f"\n  GOOD={good_avg:.4f}  BAD={bad_avg:.4f}  TRICKY={tricky_avg:.4f}  SEP={sep:.4f}")

    del model
    cleanup()


# ============================================================
# TEST 5: BGE-reranker-v2-m3
# ============================================================

def test_bge_reranker():
    from sentence_transformers import CrossEncoder

    print("\n" + "="*70)
    print("TEST: BAAI/bge-reranker-v2-m3 (purpose-built reranker)")
    print("="*70)

    try:
        model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        pairs = [(a, b) for _, a, b in ALL_PAIRS]
        raw = model.predict(pairs)
        raw = np.array(raw)

        # Normalize via sigmoid
        scores = 1 / (1 + np.exp(-raw))

        cat_scores = {"GOOD": [], "BAD": [], "TRICKY": []}
        for i, (cat, a, b) in enumerate(ALL_PAIRS):
            s = float(scores[i])
            cat_scores[cat].append(s)
            print(f"  [{cat:6s}] {s:.4f}  {a[:50]}... / {b[:50]}...")

        good_avg = np.mean(cat_scores["GOOD"])
        bad_avg = np.mean(cat_scores["BAD"])
        tricky_avg = np.mean(cat_scores["TRICKY"])
        sep = good_avg - max(bad_avg, tricky_avg)
        print(f"\n  GOOD={good_avg:.4f}  BAD={bad_avg:.4f}  TRICKY={tricky_avg:.4f}  SEP={sep:.4f}")

        del model
        cleanup()
    except Exception as e:
        print(f"  ERROR: {e}")


# ============================================================
# TEST 6: NLI contradiction score as negative signal
# ============================================================

def test_nli_contradiction_hybrid():
    """Use NLI contradiction to PENALIZE same-topic-different-argument pairs."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("\n" + "="*70)
    print("HYBRID: BGE cosine * (1 - NLI_contradiction)")
    print("="*70)

    # Get BGE scores
    bi_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    bi_scores = []
    for cat, a, b in ALL_PAIRS:
        ea = bi_model.encode("Represent this sentence: " + a, normalize_embeddings=True)
        eb = bi_model.encode(b, normalize_embeddings=True)
        bi_scores.append(cosine_sim(ea, eb))
    del bi_model
    cleanup()

    # Get NLI scores
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
    pairs = [(a, b) for _, a, b in ALL_PAIRS]
    raw = nli_model.predict(pairs)
    raw = np.array(raw)
    del nli_model
    cleanup()

    print(f"{'Cat':>7} {'BiEnc':>7} {'Contra':>7} {'Entail':>7} {'Hybrid1':>8} {'Hybrid2':>8}")
    print("-" * 60)

    cat_h1 = {"GOOD": [], "BAD": [], "TRICKY": []}
    cat_h2 = {"GOOD": [], "BAD": [], "TRICKY": []}

    for i, (cat, a, b) in enumerate(ALL_PAIRS):
        bi = bi_scores[i]
        probs = np.exp(raw[i]) / np.exp(raw[i]).sum()
        contra, neutral, entail = probs

        # Hybrid 1: cosine * (1 - contradiction)
        h1 = bi * (1 - contra)
        # Hybrid 2: cosine * entailment (from NLI)
        h2 = bi * entail

        cat_h1[cat].append(h1)
        cat_h2[cat].append(h2)

        print(f"[{cat:>6}] {bi:>7.4f} {contra:>7.4f} {entail:>7.4f} {h1:>8.4f} {h2:>8.4f}")
        print(f"         S: {a[:55]}")
        print(f"         M: {b[:55]}")

    for name, cat_scores in [("Hybrid1 (cos*(1-contra))", cat_h1), ("Hybrid2 (cos*entail)", cat_h2)]:
        good_avg = np.mean(cat_scores["GOOD"])
        bad_avg = np.mean(cat_scores["BAD"])
        tricky_avg = np.mean(cat_scores["TRICKY"])
        sep = good_avg - max(bad_avg, tricky_avg)
        print(f"\n  {name}: GOOD={good_avg:.4f}  BAD={bad_avg:.4f}  TRICKY={tricky_avg:.4f}  SEP={sep:.4f}")


if __name__ == "__main__":
    test_nli_full_breakdown()
    test_stsb_detailed()
    test_all_mpnet()
    test_bge_reranker()
    test_two_stage_pipeline()
    test_nli_contradiction_hybrid()

    print("\n\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    print("""
KEY FINDINGS TO LOOK FOR:
1. Does STSB cross-encoder cleanly separate all pairs?
2. Does the two-stage pipeline improve on either alone?
3. Does NLI contradiction score help penalize TRICKY pairs?
4. Does BGE-reranker-v2-m3 outperform STSB?
""")
