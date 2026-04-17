"""
generate_explorer_data.py — Pre-compute explorer data for the React frontend
=============================================================================
Loads Soviet + modern corpora, embeds both, selects 50 diverse passages from
each corpus, finds top-10 cross-corpus matches for each, and outputs
frontend/public/explorer_data.json.

Usage:
    python generate_explorer_data.py
    python generate_explorer_data.py --max-modern 2000 --soviet-count 50 --modern-count 50
"""
import json
import os
import sys
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np

from embedding_pipeline import CONFIG, TROPE_TAXONOMY, Passage, CorpusProcessor, EmbeddingEngine
from generate_viz_data import (
    load_soviet_corpus,
    load_modern_corpus,
    clean_modern_text,
    trim_passage,
    detect_techniques,
    compute_technique_overlap,
    extract_claims_llm,
    compute_claim_similarity,
    load_claims_cache,
    save_claims_cache,
)


def select_diverse_soviet(passages, count=50):
    """Select diverse Soviet passages spread across tropes and sources,
    prioritizing shorter/punchier ones."""
    random.seed(42)

    # Group by source
    by_source = {}
    for p in passages:
        by_source.setdefault(p.source, []).append(p)

    # Score each passage: prefer shorter, punchier text
    def punchiness(p):
        words = len(p.text.split())
        # Sweet spot: 15-40 words. Penalize very long or very short.
        length_score = max(0, 1.0 - abs(words - 30) / 40.0)
        # Boost passages with more tropes
        trope_score = min(len(p.trope_labels) * 0.2, 0.6)
        return length_score + trope_score

    # Sort each source's passages by punchiness
    for src in by_source:
        by_source[src].sort(key=punchiness, reverse=True)

    # Round-robin across sources, then across tropes
    selected = []
    selected_ids = set()

    # Phase 1: Ensure trope coverage — pick best passage per trope
    trope_to_passages = {}
    for p in passages:
        for t in p.trope_labels:
            trope_to_passages.setdefault(t, []).append(p)

    for trope in sorted(trope_to_passages.keys()):
        candidates = sorted(trope_to_passages[trope], key=punchiness, reverse=True)
        for c in candidates:
            if c.id not in selected_ids:
                selected.append(c)
                selected_ids.add(c.id)
                break
        if len(selected) >= count:
            break

    # Phase 2: Round-robin across sources
    source_iters = {src: iter(ps) for src, ps in by_source.items()}
    while len(selected) < count and source_iters:
        empty = []
        for src in list(source_iters.keys()):
            if len(selected) >= count:
                break
            for p in source_iters[src]:
                if p.id not in selected_ids:
                    selected.append(p)
                    selected_ids.add(p.id)
                    break
            else:
                empty.append(src)
        for src in empty:
            del source_iters[src]

    return selected[:count]


def select_diverse_modern(passages, processor, count=50):
    """Select diverse modern passages spread across sources,
    preferring ones with trope labels and reasonable length."""
    random.seed(42)

    # Classify tropes for scoring
    scored = []
    for p in passages:
        text = clean_modern_text(p["text"])
        words = len(text.split())
        if words < 8:
            continue  # skip too-short
        temp = Passage(
            id="tmp", text=text, source="m", source_title="",
            author="", year=p.get("year", 2024), language="en", corpus="modern"
        )
        tropes = processor.classify_tropes_keyword(temp)
        # Score: prefer medium length with trope overlap
        length_score = max(0, 1.0 - abs(words - 25) / 35.0)
        trope_score = min(len(tropes) * 0.3, 0.9)
        scored.append((p, length_score + trope_score, tropes))

    scored.sort(key=lambda x: -x[1])

    # Group by source
    by_source = {}
    for item in scored:
        src = item[0].get("source", "unknown")
        by_source.setdefault(src, []).append(item)

    # Round-robin across sources
    selected = []
    seen_texts = set()
    source_iters = {src: iter(items) for src, items in by_source.items()}

    while len(selected) < count and source_iters:
        empty = []
        for src in list(source_iters.keys()):
            if len(selected) >= count:
                break
            for item in source_iters[src]:
                norm = clean_modern_text(item[0]["text"])[:100].lower().strip()
                if norm not in seen_texts:
                    selected.append(item)
                    seen_texts.add(norm)
                    break
            else:
                empty.append(src)
        for src in empty:
            del source_iters[src]

    return selected[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate explorer data for React frontend")
    parser.add_argument("--max-modern", type=int, default=2000, help="Max modern passages to load")
    parser.add_argument("--soviet-count", type=int, default=50, help="Number of Soviet passages to select")
    parser.add_argument("--modern-count", type=int, default=50, help="Number of modern passages to select")
    parser.add_argument("--top-k", type=int, default=10, help="Number of cross-corpus matches per passage")
    parser.add_argument("--output", type=str, default="frontend/public/explorer_data.json", help="Output path")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name")
    parser.add_argument("--enrich", action="store_true", help="Add technique detection and claim extraction to explorer matches")
    args = parser.parse_args()

    model_name = args.model or CONFIG["default_model"]
    print("=" * 60)
    print("GENERATING EXPLORER DATA")
    print(f"  Model: {model_name}")
    print(f"  Soviet count: {args.soviet_count}, Modern count: {args.modern_count}")
    print(f"  Top-K matches: {args.top_k}")
    print("=" * 60)

    # --- Load corpora ---
    print("\n--- Soviet corpus ---")
    soviet_passages, processor = load_soviet_corpus()
    all_soviet_count = len(soviet_passages)
    # Filter to passages with tropes (propaganda gate)
    soviet_passages = [p for p in soviet_passages if p.trope_labels]
    print(f"  {all_soviet_count} total -> {len(soviet_passages)} with tropes")

    print("\n--- Modern corpus ---")
    modern_raw = load_modern_corpus(args.max_modern)
    print(f"  {len(modern_raw)} passages loaded")

    # --- Select diverse subsets ---
    print("\n--- Selecting diverse Soviet passages ---")
    soviet_selected = select_diverse_soviet(soviet_passages, args.soviet_count)
    print(f"  Selected {len(soviet_selected)} Soviet passages")
    src_counts = Counter(p.source for p in soviet_selected)
    print(f"  Sources: {dict(src_counts)}")
    trope_counts = Counter(t for p in soviet_selected for t in p.trope_labels)
    print(f"  Trope coverage: {dict(trope_counts)}")

    print("\n--- Selecting diverse modern passages ---")
    modern_selected = select_diverse_modern(modern_raw, processor, args.modern_count)
    print(f"  Selected {len(modern_selected)} modern passages")
    mod_src_counts = Counter(item[0].get("source", "?") for item in modern_selected)
    print(f"  Sources: {dict(mod_src_counts)}")

    # --- Embed everything ---
    print("\n--- Embedding ---")
    engine = EmbeddingEngine(model_name)
    prefix = "query: " if "e5" in model_name.lower() else ""

    # Embed ALL soviet and modern for matching
    print("  Embedding full Soviet corpus...")
    soviet_passages = engine.embed_passages(soviet_passages)
    soviet_matrix = np.array([p.embedding for p in soviet_passages])

    print("  Embedding full modern corpus...")
    modern_texts_clean = [clean_modern_text(p["text"]) for p in modern_raw]
    modern_embeddings = engine.model.encode(
        [prefix + t for t in modern_texts_clean],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    # Build index maps for selected passages
    soviet_selected_ids = {p.id for p in soviet_selected}
    soviet_idx_map = {p.id: i for i, p in enumerate(soviet_passages)}

    # For modern selected, we need their indices in modern_raw
    modern_selected_indices = []
    for item in modern_selected:
        orig_passage = item[0]
        # Find index in modern_raw
        for i, mr in enumerate(modern_raw):
            if mr is orig_passage:
                modern_selected_indices.append(i)
                break

    # --- Compute cross-corpus similarities ---
    print("\n--- Computing cross-corpus matches ---")

    # Soviet -> Modern: for each selected Soviet passage, find top-K modern matches
    soviet_results = []
    for sp in soviet_selected:
        idx = soviet_idx_map[sp.id]
        sov_emb = soviet_matrix[idx]
        # Compute similarity to all modern passages
        sims = np.dot(modern_embeddings, sov_emb)
        top_indices = np.argsort(sims)[::-1][:args.top_k]

        top_matches = []
        for mi in top_indices:
            mod_text = modern_texts_clean[mi]
            mod_raw = modern_raw[mi]
            top_matches.append({
                "text": trim_passage(mod_text),
                "source": mod_raw.get("source_title", mod_raw.get("source", "")),
                "year": mod_raw.get("year", 2024),
                "similarity": round(float(sims[mi]), 4),
                "corpus": "modern",
            })

        soviet_results.append({
            "id": f"s_{len(soviet_results)}",
            "text": trim_passage(sp.text),
            "textFull": sp.text,
            "source": sp.source_title,
            "year": sp.year,
            "tropes": sp.trope_labels,
            "top_matches": top_matches,
        })

    # Modern -> Soviet: for each selected modern passage, find top-K soviet matches
    modern_results = []
    for item_idx, mod_idx in enumerate(modern_selected_indices):
        mod_emb = modern_embeddings[mod_idx]
        mod_raw = modern_raw[mod_idx]
        mod_text = modern_texts_clean[mod_idx]

        # Compute similarity to all soviet passages
        sims = np.dot(soviet_matrix, mod_emb)
        top_indices = np.argsort(sims)[::-1][:args.top_k]

        top_matches = []
        for si in top_indices:
            sp = soviet_passages[si]
            top_matches.append({
                "text": trim_passage(sp.text),
                "source": sp.source_title,
                "year": sp.year,
                "similarity": round(float(sims[si]), 4),
                "corpus": "soviet",
                "tropes": sp.trope_labels,
            })

        modern_results.append({
            "id": f"m_{item_idx}",
            "text": trim_passage(mod_text),
            "textFull": mod_text,
            "source": mod_raw.get("source_title", mod_raw.get("source", "")),
            "year": mod_raw.get("year", 2024),
            "top_matches": top_matches,
        })

    # --- Enrichment: technique detection + claim extraction ---
    if args.enrich:
        print("\n--- Enriching explorer matches with techniques ---")
        try:
            from transformers import pipeline as hf_pipeline
            tech_classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

            # Detect techniques for selected passages
            print("  Soviet passage techniques...")
            for sr in soviet_results:
                sr["techniques"] = list(detect_techniques(tech_classifier, sr["textFull"]).keys())

            print("  Modern passage techniques...")
            for mr in modern_results:
                mr["techniques"] = list(detect_techniques(tech_classifier, mr["textFull"]).keys())

            # Detect techniques for top matches and compute overlap
            print("  Match technique overlap...")
            for sr in soviet_results:
                sov_techs = detect_techniques(tech_classifier, sr["textFull"])
                for match in sr["top_matches"]:
                    mod_techs = detect_techniques(tech_classifier, match["text"])
                    overlap, shared = compute_technique_overlap(sov_techs, mod_techs)
                    match["sharedTechniques"] = shared
                    match["techniqueOverlap"] = round(overlap, 4)

            for mr in modern_results:
                mod_techs = detect_techniques(tech_classifier, mr["textFull"])
                for match in mr["top_matches"]:
                    sov_techs = detect_techniques(tech_classifier, match["text"])
                    overlap, shared = compute_technique_overlap(mod_techs, sov_techs)
                    match["sharedTechniques"] = shared
                    match["techniqueOverlap"] = round(overlap, 4)

            print("  Done.")
        except Exception as e:
            print(f"  WARNING: Technique detection failed: {e}")

        # Claim extraction for top-3 matches per passage
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print("\n--- Enriching explorer matches with claims ---")
            try:
                from anthropic import Anthropic
                client = Anthropic()
                claims_cache = load_claims_cache()

                for sr in soviet_results:
                    sov_key = f"soviet:{sr['textFull'][:200]}"
                    sov_claims = claims_cache.get(sov_key) or extract_claims_llm(sr["textFull"], "propaganda", client)
                    claims_cache[sov_key] = sov_claims
                    sr["claims"] = sov_claims

                    for match in sr["top_matches"][:3]:
                        mod_key = f"modern:{match['text'][:200]}"
                        mod_claims = claims_cache.get(mod_key) or extract_claims_llm(match["text"], "echo", client)
                        claims_cache[mod_key] = mod_claims
                        if sov_claims and mod_claims:
                            score, best_sov, best_mod = compute_claim_similarity(engine, sov_claims, mod_claims)
                            match["claimSimilarity"] = round(score, 4)
                            match["claimPair"] = {"sourceClaim": best_sov, "matchClaim": best_mod}

                for mr in modern_results:
                    mod_key = f"modern:{mr['textFull'][:200]}"
                    mod_claims = claims_cache.get(mod_key) or extract_claims_llm(mr["textFull"], "echo", client)
                    claims_cache[mod_key] = mod_claims
                    mr["claims"] = mod_claims

                    for match in mr["top_matches"][:3]:
                        sov_key = f"soviet:{match['text'][:200]}"
                        sov_claims = claims_cache.get(sov_key) or extract_claims_llm(match["text"], "propaganda", client)
                        claims_cache[sov_key] = sov_claims
                        if mod_claims and sov_claims:
                            score, best_mod, best_sov = compute_claim_similarity(engine, mod_claims, sov_claims)
                            match["claimSimilarity"] = round(score, 4)
                            match["claimPair"] = {"sourceClaim": best_mod, "matchClaim": best_sov}

                save_claims_cache(claims_cache)
                print(f"  Claims cache saved ({len(claims_cache)} entries)")
            except Exception as e:
                print(f"  WARNING: Claim extraction failed: {e}")

        # Compute ensemble scores and re-sort matches
        print("\n--- Computing ensemble scores for explorer matches ---")
        for results in [soviet_results, modern_results]:
            for passage in results:
                for match in passage["top_matches"]:
                    claim = match.get("claimSimilarity", match["similarity"])
                    tech = match.get("techniqueOverlap", 0.0)
                    match["ensembleScore"] = round(claim * 0.6 + tech * 0.4, 4)
                # Re-sort by ensemble score (highest first)
                passage["top_matches"].sort(key=lambda m: m.get("ensembleScore", m["similarity"]), reverse=True)
        print("  Done.")

    # --- Output ---
    output = {
        "soviet_passages": soviet_results,
        "modern_passages": modern_results,
    }

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_path)
    print(f"\n--- Done ---")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    print(f"  Soviet passages: {len(soviet_results)}")
    print(f"  Modern passages: {len(modern_results)}")
    print(f"  Total match pairs: {(len(soviet_results) + len(modern_results)) * args.top_k}")


if __name__ == "__main__":
    main()
