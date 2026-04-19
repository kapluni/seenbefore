"""
Run statistical quality metrics against the current pipeline and corpora.
Outputs BOTH the markdown summary AND a stats_data.json with raw score arrays
for use by the visualization scripts.

Standalone: not part of the production pipeline, lives in viz_examples/.
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from generate_viz_data import LEGITIMATE_TEXTS, SAMPLE_MODERN, load_soviet_corpus, load_modern_corpus
from embedding_pipeline import EmbeddingEngine, CONFIG


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled else 0.0


def bootstrap_ci(x, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(0)
    means = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, (1-ci)/2*100)), float(np.percentile(means, (1+ci)/2*100))


def permutation_test(observed_scores, soviet_embs, modern_embs, n_perms=10000):
    n = len(observed_scores)
    rng = np.random.default_rng(1)
    perm_means = []
    for _ in range(n_perms):
        si = rng.choice(len(soviet_embs), size=n, replace=False)
        mi = rng.choice(len(modern_embs), size=n, replace=False)
        sims = [float(np.dot(soviet_embs[s], modern_embs[m])) for s, m in zip(si, mi)]
        perm_means.append(float(np.mean(sims)))
    perm_means = np.array(perm_means)
    obs = float(np.mean(observed_scores))
    return float(np.mean(perm_means >= obs)), perm_means


def main():
    print("Loading model + corpora...")
    engine = EmbeddingEngine(CONFIG["default_model"])
    engine.load_model()

    soviet_passages, _ = load_soviet_corpus()
    soviet_passages = [p for p in soviet_passages if p.trope_labels]
    soviet_passages = engine.embed_passages(soviet_passages)
    soviet_matrix = np.array([p.embedding for p in soviet_passages])
    print(f"Soviet: {len(soviet_passages)} passages")

    modern_raw = load_modern_corpus(max_passages=2000)
    modern_texts = [p["text"] for p in modern_raw]
    modern_embs = engine.model.encode(
        modern_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"Modern: {len(modern_embs)} passages")

    with open("viz_data.json") as f:
        viz = json.load(f)
    matches = viz["matches"]
    match_scores = np.array([m["similarity"] for m in matches])
    match_ensemble = np.array([m.get("ensembleScore", m["similarity"]) for m in matches])
    print(f"Curated matches: n={len(matches)} mean cosine={np.mean(match_scores):.4f} mean ensemble={np.mean(match_ensemble):.4f}")

    legit_embs = engine.model.encode([t["text"] for t in LEGITIMATE_TEXTS], normalize_embeddings=True)
    legit_scores = np.array([float(np.max(np.dot(soviet_matrix, e))) for e in legit_embs])

    echo_embs = engine.model.encode([t["text"] for t in SAMPLE_MODERN], normalize_embeddings=True)
    echo_scores = np.array([float(np.max(np.dot(soviet_matrix, e))) for e in echo_embs])

    rng = np.random.default_rng(42)
    n_random = 1000
    rs = rng.choice(len(soviet_matrix), size=n_random)
    rm = rng.choice(len(modern_embs), size=n_random)
    random_scores = np.array([float(np.dot(soviet_matrix[s], modern_embs[m])) for s, m in zip(rs, rm)])

    p_perm, perm_means = permutation_test(match_scores, soviet_matrix, modern_embs, n_perms=10000)
    d_match_random = cohens_d(match_scores, random_scores)
    d_match_legit = cohens_d(match_scores, legit_scores)
    d_echo_legit = cohens_d(echo_scores, legit_scores)
    d_echo_random = cohens_d(echo_scores, random_scores)

    ci_match = bootstrap_ci(match_scores)
    ci_echo = bootstrap_ci(echo_scores)
    ci_legit = bootstrap_ci(legit_scores)
    ci_random = bootstrap_ci(random_scores)

    y_true = np.concatenate([np.ones(len(match_scores)), np.zeros(len(random_scores))])
    y_scores = np.concatenate([match_scores, random_scores])
    auc_match_random = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    opt_idx = int(np.argmax(tpr - fpr))
    opt_threshold = float(thr[opt_idx])

    y_true2 = np.concatenate([np.ones(len(echo_scores)), np.zeros(len(legit_scores))])
    y_scores2 = np.concatenate([echo_scores, legit_scores])
    auc_echo_legit = float(roc_auc_score(y_true2, y_scores2))

    all_pos = np.concatenate([match_scores, echo_scores])
    all_neg = np.concatenate([legit_scores, random_scores])
    y_true3 = np.concatenate([np.ones(len(all_pos)), np.zeros(len(all_neg))])
    y_scores3 = np.concatenate([all_pos, all_neg])
    auc_all = float(roc_auc_score(y_true3, y_scores3))

    u1, p1 = stats.mannwhitneyu(match_scores, random_scores, alternative="greater")
    u2, p2 = stats.mannwhitneyu(match_scores, legit_scores, alternative="greater")
    u3, p3 = stats.mannwhitneyu(echo_scores, legit_scores, alternative="greater")
    u4, p4 = stats.mannwhitneyu(echo_scores, random_scores, alternative="greater")

    out = {
        "n": {
            "matches": int(len(match_scores)),
            "echoes": int(len(echo_scores)),
            "legit": int(len(legit_scores)),
            "random": int(len(random_scores)),
            "soviet_corpus": int(len(soviet_matrix)),
            "modern_corpus": int(len(modern_embs)),
        },
        "scores": {
            "match_cosine": match_scores.tolist(),
            "match_ensemble": match_ensemble.tolist(),
            "echo": echo_scores.tolist(),
            "legit": legit_scores.tolist(),
            "random": random_scores.tolist(),
        },
        "distributions": {
            "curated_matches": {
                "mean": float(np.mean(match_scores)), "std": float(np.std(match_scores)),
                "min": float(np.min(match_scores)), "median": float(np.median(match_scores)),
                "max": float(np.max(match_scores)),
            },
            "known_echoes": {
                "mean": float(np.mean(echo_scores)), "std": float(np.std(echo_scores)),
                "min": float(np.min(echo_scores)), "median": float(np.median(echo_scores)),
                "max": float(np.max(echo_scores)),
            },
            "legit_criticism": {
                "mean": float(np.mean(legit_scores)), "std": float(np.std(legit_scores)),
                "min": float(np.min(legit_scores)), "median": float(np.median(legit_scores)),
                "max": float(np.max(legit_scores)),
            },
            "random_pairs": {
                "mean": float(np.mean(random_scores)), "std": float(np.std(random_scores)),
                "min": float(np.min(random_scores)), "median": float(np.median(random_scores)),
                "max": float(np.max(random_scores)),
            },
        },
        "permutation": {
            "observed_mean": float(np.mean(match_scores)),
            "null_mean": float(np.mean(perm_means)),
            "null_std": float(np.std(perm_means)),
            "p_value": p_perm,
            "n_perms": 10000,
        },
        "cohens_d": {
            "matches_vs_random": float(d_match_random),
            "matches_vs_legit": float(d_match_legit),
            "echoes_vs_legit": float(d_echo_legit),
            "echoes_vs_random": float(d_echo_random),
        },
        "bootstrap_ci_95": {
            "curated_matches": list(ci_match),
            "known_echoes": list(ci_echo),
            "legit_criticism": list(ci_legit),
            "random_pairs": list(ci_random),
        },
        "auc": {
            "matches_vs_random": auc_match_random,
            "echoes_vs_legit": auc_echo_legit,
            "all_positives_vs_all_negatives": auc_all,
            "optimal_threshold_youden": opt_threshold,
        },
        "mann_whitney_u": {
            "matches_vs_random": {"U": float(u1), "p": float(p1)},
            "matches_vs_legit": {"U": float(u2), "p": float(p2)},
            "echoes_vs_legit": {"U": float(u3), "p": float(p3)},
            "echoes_vs_random": {"U": float(u4), "p": float(p4)},
        },
    }

    with open("viz_examples/stats_data.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote viz_examples/stats_data.json")
    print(f"AUC matches-vs-random: {auc_match_random:.4f}")
    print(f"Cohen's d matches-vs-random: {d_match_random:.3f}")
    print(f"Permutation p: {p_perm}")
    print(f"Youden threshold: {opt_threshold:.4f}")


if __name__ == "__main__":
    main()
