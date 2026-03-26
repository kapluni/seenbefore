#!/usr/bin/env python3
"""
Benchmark embedding models for argument-aware matching.

Tests cross-encoders, instruction-tuned models, LLM-based embeddings,
and NLI-trained models on Soviet propaganda -> modern rhetoric pairs.

Specifically tests separation between:
  GOOD pairs (same argument, should score HIGH)
  BAD pairs (same topic different argument, should score LOW)
  TRICKY pairs (same topic opposite argument, should score LOW)
"""

import time
import sys
import numpy as np
import torch

# ============================================================
# TEST PAIRS
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


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_scores(model_name, results):
    """Pretty-print scores with category averages."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    cat_scores = {"GOOD": [], "BAD": [], "TRICKY": []}
    for cat, a, b, score in results:
        cat_scores[cat].append(score)
        marker = ""
        if cat == "GOOD" and score < 0.5:
            marker = " << TOO LOW"
        elif cat == "BAD" and score > 0.5:
            marker = " << TOO HIGH"
        elif cat == "TRICKY" and score > 0.5:
            marker = " << TOO HIGH"
        print(f"  [{cat:6s}] {score:.4f}{marker}")
        print(f"           S: {a[:65]}")
        print(f"           M: {b[:65]}")

    good_avg = np.mean(cat_scores["GOOD"])
    bad_avg = np.mean(cat_scores["BAD"])
    tricky_avg = np.mean(cat_scores["TRICKY"])
    separation = good_avg - max(bad_avg, tricky_avg)

    print(f"\n  AVERAGES:  GOOD={good_avg:.4f}  BAD={bad_avg:.4f}  TRICKY={tricky_avg:.4f}")
    print(f"  SEPARATION (GOOD_avg - max(BAD_avg, TRICKY_avg)): {separation:.4f}")
    grade = 'EXCELLENT' if separation > 0.3 else 'GOOD' if separation > 0.2 else 'FAIR' if separation > 0.1 else 'POOR'
    print(f"  GRADE: {grade}")

    return {"good": good_avg, "bad": bad_avg, "tricky": tricky_avg, "separation": separation}


def cleanup():
    """Free GPU/MPS memory."""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ============================================================
# BI-ENCODER TESTING
# ============================================================

def test_bi_encoder(model_name, query_prefix="", passage_prefix="", prompt=None, trust_remote_code=False):
    """Test a bi-encoder model."""
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    print(f"  Loaded in {time.time()-t0:.1f}s, dim={model.get_sentence_embedding_dimension()}")

    results = []
    for cat, a, b in ALL_PAIRS:
        if prompt:
            emb_a = model.encode(a, prompt=prompt, normalize_embeddings=True)
            emb_b = model.encode(b, prompt=prompt, normalize_embeddings=True)
        else:
            emb_a = model.encode(query_prefix + a, normalize_embeddings=True)
            emb_b = model.encode(passage_prefix + b, normalize_embeddings=True)
        sim = cosine_sim(emb_a, emb_b)
        results.append((cat, a, b, sim))

    del model
    cleanup()
    return results


# ============================================================
# CROSS-ENCODER TESTING
# ============================================================

def test_cross_encoder(model_name):
    """Test a cross-encoder (scores pairs directly, no embeddings)."""
    from sentence_transformers import CrossEncoder

    print(f"\nLoading cross-encoder {model_name}...")
    t0 = time.time()
    model = CrossEncoder(model_name, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    pairs = [(a, b) for _, a, b in ALL_PAIRS]
    raw_scores = model.predict(pairs)

    # Normalize to [0,1] via sigmoid if needed
    raw_scores = np.array(raw_scores)
    if raw_scores.min() < 0 or raw_scores.max() > 1:
        scores = 1 / (1 + np.exp(-raw_scores))
    else:
        scores = raw_scores

    results = []
    for i, (cat, a, b) in enumerate(ALL_PAIRS):
        results.append((cat, a, b, float(scores[i]) if scores.ndim == 1 else float(scores[i])))

    del model
    cleanup()
    return results


# ============================================================
# NLI CROSS-ENCODER (entailment probability)
# ============================================================

def test_nli_cross_encoder(model_name):
    """Test NLI cross-encoder using entailment probability as similarity."""
    from sentence_transformers import CrossEncoder

    print(f"\nLoading NLI cross-encoder {model_name}...")
    t0 = time.time()
    model = CrossEncoder(model_name, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    pairs = [(a, b) for _, a, b in ALL_PAIRS]
    raw_scores = model.predict(pairs)
    raw_scores = np.array(raw_scores)

    results = []
    for i, (cat, a, b) in enumerate(ALL_PAIRS):
        if raw_scores.ndim > 1 and raw_scores.shape[1] == 3:
            # [contradiction, neutral, entailment]
            probs = np.exp(raw_scores[i]) / np.exp(raw_scores[i]).sum()
            score = float(probs[2])  # entailment prob
        else:
            score = float(raw_scores[i])
            if score < 0 or score > 1:
                score = 1 / (1 + np.exp(-score))
        results.append((cat, a, b, score))

    del model
    cleanup()
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    summary = {}

    # ----------------------------------------------------------
    # APPROACH 0: BASELINE
    # ----------------------------------------------------------
    print("\n" + "#"*70)
    print("# APPROACH 0: BASELINE (current model)")
    print("#"*70)

    results = test_bi_encoder(
        "BAAI/bge-large-en-v1.5",
        query_prefix="Represent this sentence: ",
    )
    summary["BGE-large-en-v1.5 (baseline)"] = print_scores(
        "BAAI/bge-large-en-v1.5 (baseline)", results)

    # ----------------------------------------------------------
    # APPROACH 1: CROSS-ENCODERS FOR RERANKING
    # ----------------------------------------------------------
    print("\n" + "#"*70)
    print("# APPROACH 1: CROSS-ENCODERS")
    print("#"*70)

    for ce_model in [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/stsb-roberta-large",
    ]:
        try:
            results = test_cross_encoder(ce_model)
            summary[ce_model] = print_scores(ce_model, results)
        except Exception as e:
            print(f"  ERROR with {ce_model}: {e}")

    # ----------------------------------------------------------
    # APPROACH 2: INSTRUCTION-TUNED EMBEDDINGS
    # ----------------------------------------------------------
    print("\n" + "#"*70)
    print("# APPROACH 2: INSTRUCTION-TUNED EMBEDDINGS")
    print("#"*70)

    # BGE with custom propaganda instruction
    results = test_bi_encoder(
        "BAAI/bge-large-en-v1.5",
        query_prefix="Represent this propaganda claim for finding modern texts making the same argument: ",
    )
    summary["BGE + propaganda instruction"] = print_scores(
        "BGE-large + propaganda instruction", results)

    # E5-large-v2
    try:
        results = test_bi_encoder(
            "intfloat/e5-large-v2",
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
        summary["E5-large-v2"] = print_scores("intfloat/e5-large-v2", results)
    except Exception as e:
        print(f"  ERROR with E5-large-v2: {e}")

    # Multilingual E5 large
    try:
        results = test_bi_encoder(
            "intfloat/multilingual-e5-large",
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
        summary["multilingual-e5-large"] = print_scores("intfloat/multilingual-e5-large", results)
    except Exception as e:
        print(f"  ERROR: {e}")

    # ----------------------------------------------------------
    # APPROACH 3: LLM-BASED EMBEDDINGS
    # ----------------------------------------------------------
    print("\n" + "#"*70)
    print("# APPROACH 3: LLM-BASED / LARGER EMBEDDINGS")
    print("#"*70)

    # GTE-large (Alibaba, smaller variant that fits in memory)
    try:
        results = test_bi_encoder("thenlper/gte-large")
        summary["GTE-large"] = print_scores("thenlper/gte-large", results)
    except Exception as e:
        print(f"  ERROR with GTE-large: {e}")

    # Jina embeddings v3
    try:
        results = test_bi_encoder("jinaai/jina-embeddings-v3", trust_remote_code=True)
        summary["jina-embeddings-v3"] = print_scores("jinaai/jina-embeddings-v3", results)
    except Exception as e:
        print(f"  ERROR with jina-v3: {e}")

    # Nomic embed v1.5
    try:
        results = test_bi_encoder(
            "nomic-ai/nomic-embed-text-v1.5",
            query_prefix="search_query: ",
            passage_prefix="search_document: ",
            trust_remote_code=True,
        )
        summary["nomic-embed-v1.5"] = print_scores("nomic-ai/nomic-embed-text-v1.5", results)
    except Exception as e:
        print(f"  ERROR with nomic: {e}")

    # ----------------------------------------------------------
    # APPROACH 4: NLI-TRAINED MODELS
    # ----------------------------------------------------------
    print("\n" + "#"*70)
    print("# APPROACH 4: NLI-TRAINED MODELS (entailment scoring)")
    print("#"*70)

    for nli_model in [
        "cross-encoder/nli-deberta-v3-large",
        "cross-encoder/nli-deberta-v3-base",
    ]:
        try:
            results = test_nli_cross_encoder(nli_model)
            summary[nli_model] = print_scores(nli_model + " (entailment)", results)
        except Exception as e:
            print(f"  ERROR with {nli_model}: {e}")

    # NLI bi-encoder
    try:
        results = test_bi_encoder("sentence-transformers/nli-mpnet-base-v2")
        summary["nli-mpnet-base-v2"] = print_scores("nli-mpnet-base-v2", results)
    except Exception as e:
        print(f"  ERROR with nli-mpnet: {e}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n\n" + "="*80)
    print("FINAL RANKING BY SEPARATION (GOOD_avg - max(BAD_avg, TRICKY_avg))")
    print("="*80)
    print(f"{'Model':<45} {'GOOD':>7} {'BAD':>7} {'TRICKY':>7} {'SEP':>7} {'Type'}")
    print("-" * 85)

    # Classify model types
    ce_models = {"cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/stsb-roberta-large",
                 "cross-encoder/nli-deberta-v3-large", "cross-encoder/nli-deberta-v3-base"}

    ranked = sorted(summary.items(), key=lambda x: x[1]["separation"], reverse=True)
    for name, s in ranked:
        mtype = "cross-enc" if any(ce in name for ce in ["cross-encoder", "nli-deberta"]) else "bi-enc"
        print(f"{name:<45} {s['good']:>7.4f} {s['bad']:>7.4f} {s['tricky']:>7.4f} {s['separation']:>7.4f} {mtype}")

    best_bi = [(n, s) for n, s in ranked if "cross-encoder" not in n.lower() and "nli-deberta" not in n.lower()]
    best_ce = [(n, s) for n, s in ranked if "cross-encoder" in n.lower() or "nli-deberta" in n.lower()]

    print(f"\nBest bi-encoder: {best_bi[0][0]} (sep={best_bi[0][1]['separation']:.4f})" if best_bi else "")
    print(f"Best cross-encoder: {best_ce[0][0]} (sep={best_ce[0][1]['separation']:.4f})" if best_ce else "")

    print(f"\nRECOMMENDATION:")
    print(f"  - Use best bi-encoder for initial retrieval/indexing")
    print(f"  - Use best cross-encoder/NLI model as reranker on top-K candidates")
    print(f"  - Cross-encoders score pairs directly (no embeddings), so they")
    print(f"    cannot be used for vector indexing, only for reranking.")


if __name__ == "__main__":
    main()
