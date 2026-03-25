"""
Benchmark embedding models for Soviet anti-Zionist propaganda ↔ modern rhetoric matching.
Tests separation quality between true semantic matches and non-matches.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Test pairs ---
GOOD_PAIRS = [
    ("The Zionist bourgeoisie controls international finance and media",
     "Jews control the banks and the media"),
    ("Zionism is a form of racism and racial discrimination",
     "Israel is an apartheid state practicing racial discrimination"),
    ("The Zionists collaborated with the Nazis to further their colonial project",
     "Zionists worked with the Nazis, they're two sides of the same coin"),
    ("Claims of antisemitism are weaponized to silence criticism of Israel",
     "They call everything antisemitic to shut down debate about Palestine"),
    ("Israel is an outpost of American imperialism in the Middle East",
     "Israel is a settler-colonial project serving Western imperial interests"),
]

BAD_PAIRS = [
    ("The Zionist bourgeoisie controls international finance and media",
     "I think the Israeli government should halt settlement construction"),
    ("Zionism is a form of racism and racial discrimination",
     "Both Israeli and Palestinian children deserve to grow up in peace"),
    ("The Zionists collaborated with the Nazis",
     "The GDP of Israel grew 3.4% in the last quarter"),
    ("Claims of antisemitism are weaponized to silence criticism",
     "My favorite restaurant in Tel Aviv makes amazing hummus"),
    ("Israel is an outpost of American imperialism",
     "The two-state solution requires compromises from both sides"),
]

# --- Model configs ---
MODELS = [
    {
        "name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "trust_remote_code": False,
    },
    {
        "name": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence: ",
        "passage_prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "query_prefix": "search_query: ",
        "passage_prefix": "search_document: ",
        "trust_remote_code": True,
    },
]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def benchmark_model(config):
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"Loading: {name}")
    print(f"{'='*60}")

    load_start = time.time()
    model = SentenceTransformer(name, trust_remote_code=config["trust_remote_code"])
    load_time = time.time() - load_start
    print(f"  Load time: {load_time:.1f}s")

    qp = config["query_prefix"]
    pp = config["passage_prefix"]

    # Collect all texts
    soviet_texts = [pp + p[0] for p in GOOD_PAIRS + BAD_PAIRS]
    modern_texts = [qp + p[1] for p in GOOD_PAIRS + BAD_PAIRS]

    encode_start = time.time()
    soviet_embs = model.encode(soviet_texts, normalize_embeddings=True)
    modern_embs = model.encode(modern_texts, normalize_embeddings=True)
    encode_time = time.time() - encode_start

    good_scores = []
    bad_scores = []

    print(f"\n  GOOD pairs (should be HIGH):")
    for i, (s, m) in enumerate(GOOD_PAIRS):
        score = cosine_sim(soviet_embs[i], modern_embs[i])
        good_scores.append(score)
        print(f"    {score:.4f} | {s[:50]}... -> {m[:50]}...")

    print(f"\n  BAD pairs (should be LOW):")
    for i, (s, m) in enumerate(BAD_PAIRS):
        idx = i + len(GOOD_PAIRS)
        score = cosine_sim(soviet_embs[idx], modern_embs[idx])
        bad_scores.append(score)
        print(f"    {score:.4f} | {s[:50]}... -> {m[:50]}...")

    result = {
        "name": name,
        "avg_good": np.mean(good_scores),
        "avg_bad": np.mean(bad_scores),
        "gap": np.mean(good_scores) - np.mean(bad_scores),
        "min_good": min(good_scores),
        "max_bad": max(bad_scores),
        "threshold_margin": min(good_scores) - max(bad_scores),
        "encode_time": encode_time,
        "load_time": load_time,
        "good_scores": good_scores,
        "bad_scores": bad_scores,
    }
    return result


def main():
    results = []
    for config in MODELS:
        try:
            r = benchmark_model(config)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR with {config['name']}: {e}")

    # Summary table
    print("\n\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    header = f"{'Model':<45} {'AvgGood':>8} {'AvgBad':>8} {'Gap':>8} {'MinGood':>8} {'MaxBad':>8} {'Margin':>8} {'EncTime':>8}"
    print(header)
    print("-" * 100)

    for r in results:
        line = (
            f"{r['name']:<45} "
            f"{r['avg_good']:>8.4f} "
            f"{r['avg_bad']:>8.4f} "
            f"{r['gap']:>8.4f} "
            f"{r['min_good']:>8.4f} "
            f"{r['max_bad']:>8.4f} "
            f"{r['threshold_margin']:>8.4f} "
            f"{r['encode_time']:>7.2f}s"
        )
        print(line)

    print("-" * 100)
    print("\nGap = AvgGood - AvgBad (higher is better)")
    print("Margin = MinGood - MaxBad (positive means clean threshold separation)")
    print()

    # Rank by gap
    ranked = sorted(results, key=lambda x: x["gap"], reverse=True)
    print("RANKING by Gap:")
    for i, r in enumerate(ranked, 1):
        sep = "CLEAN" if r["threshold_margin"] > 0 else f"OVERLAP by {-r['threshold_margin']:.4f}"
        print(f"  {i}. {r['name']} — Gap: {r['gap']:.4f}, Threshold: {sep}")


if __name__ == "__main__":
    main()
