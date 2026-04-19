"""
Ridgeline plot of cosine similarity distributions across four groups:
- Curated matches (n=24)
- Known echoes (hand-crafted modern text known to echo Soviet; n≈10)
- Legitimate criticism (n=8 calibration texts)
- Random Soviet-modern pairs (n=1000 null distribution)

Reads the score arrays from viz_examples/stats_data.json (produced by run_stats.py).
Draws KDE curves stacked in rows with overlap, colored to separate positives
(right-shifted) from legitimate criticism and random pairs (left-shifted).

Outputs viz_examples/ridgeline.png.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))


GROUPS = [
    ("Random Soviet–Modern pairs", "random",   "#9ca3af", "Null distribution (n=1000)"),
    ("Legitimate policy criticism", "legit",    "#27ae60", "Calibration set (n=8)"),
    ("Known echoes (hand-crafted)", "echo",     "#3498db", "Reference set (n=10)"),
    ("Curated matches (viz_data)",  "match",    "#c0392b", "Final match list (n=24)"),
]


def main():
    with open("viz_examples/stats_data.json") as f:
        S = json.load(f)
    scores = {
        "match":  np.array(S["scores"]["match_cosine"]),
        "echo":   np.array(S["scores"]["echo"]),
        "legit":  np.array(S["scores"]["legit"]),
        "random": np.array(S["scores"]["random"]),
    }
    youden = S["auc"]["optimal_threshold_youden"]
    auc_match = S["auc"]["matches_vs_random"]
    auc_echo_legit = S["auc"]["echoes_vs_legit"]
    d_match = S["cohens_d"]["matches_vs_random"]
    perm_p = S["permutation"]["p_value"]

    x_grid = np.linspace(0.2, 0.95, 400)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=140)
    n_groups = len(GROUPS)
    row_gap = 1.0

    max_density = 0.0
    curves = []
    for label, key, color, sub in GROUPS:
        data = scores[key]
        kde = gaussian_kde(data, bw_method=0.35)
        y = kde(x_grid)
        max_density = max(max_density, float(np.max(y)))
        curves.append((label, sub, color, y, data))

    scale = 0.8 / max_density

    for i, (label, sub, color, y, data) in enumerate(curves):
        base = (n_groups - 1 - i) * row_gap
        y_plot = y * scale + base
        ax.fill_between(x_grid, base, y_plot, color=color, alpha=0.55, linewidth=0)
        ax.plot(x_grid, y_plot, color=color, linewidth=1.5)
        ax.plot(data, np.full_like(data, base - 0.03), "|", color=color, markersize=10, alpha=0.8)
        mean_v = float(np.mean(data))
        ax.plot([mean_v, mean_v], [base, base + 0.9], color=color, linewidth=1.2, linestyle=":", alpha=0.9)
        ax.text(0.205, base + 0.55, label, fontsize=11, fontweight="bold", va="center", color="#111")
        ax.text(0.205, base + 0.32, sub, fontsize=9, va="center", color="#555")
        ax.text(0.94, base + 0.55, f"μ={mean_v:.3f}", fontsize=9.5, va="center", ha="right", color=color)

    ax.axvline(youden, color="#111", linewidth=1.1, linestyle="--", alpha=0.55)
    ax.text(youden, (n_groups - 0.3) * row_gap, f" Youden threshold {youden:.2f}",
            fontsize=9, color="#111", va="top")

    ax.set_xlim(0.20, 0.95)
    ax.set_ylim(-0.4, n_groups * row_gap)
    ax.set_xlabel("Cosine similarity (BGE-large-en-v1.5, best-of Soviet corpus)", fontsize=11)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Distribution of similarity scores across four reference groups",
                 fontsize=13, fontweight="bold", loc="left", pad=14, fontname="Georgia")
    subtitle = (f"AUC matches vs random = {auc_match:.3f}  ·  "
                f"AUC echoes vs legit criticism = {auc_echo_legit:.3f}  ·  "
                f"Cohen's d = {d_match:.2f}  ·  permutation p = {perm_p if perm_p>0 else '<1e-4'}")
    ax.text(0.02, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color="#444",
            fontname="Georgia")

    fig.tight_layout()
    out = "viz_examples/ridgeline.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
