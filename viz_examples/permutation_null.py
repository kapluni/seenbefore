"""
Permutation-null histogram: a single picture that communicates p < 1e-4.

Plots the distribution of 1,000 random Soviet-modern cosine similarities
(the null distribution), with the observed mean of the 24 curated matches
drawn as a thick vertical line far to the right of the bulk. Small tick
marks above the histogram show where the 8 legitimate-criticism and 10
known-echo scores sit, so the reader can see the harder comparisons.

Reference style: sci-kit learn permutation-test plot and Jared Wilber's
permutation explainer.

Outputs viz_examples/permutation_null.png.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))


def main():
    with open("viz_examples/stats_data.json") as f:
        S = json.load(f)

    random_scores = np.array(S["scores"]["random"])
    match_scores = np.array(S["scores"]["match_cosine"])
    echo_scores = np.array(S["scores"]["echo"])
    legit_scores = np.array(S["scores"]["legit"])

    observed = float(np.mean(match_scores))
    null_mean = S["permutation"]["null_mean"]
    null_std = S["permutation"]["null_std"]
    n_perms = S["permutation"]["n_perms"]
    cohens_d = S["cohens_d"]["matches_vs_random"]
    auc_mvr = S["auc"]["matches_vs_random"]

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=160)

    # Histogram of random pair cosines
    bins = np.linspace(0.25, 0.95, 41)
    counts, edges, patches = ax.hist(
        random_scores, bins=bins, color="#e6e6e6", edgecolor="#555",
        linewidth=0.6, zorder=2,
    )
    y_hist_max = float(counts.max())

    # Shaded tail region beyond the observed statistic.
    # Use counts from a "tail mask" so the fill is consistent with the bars.
    tail_edges = [e for e in edges if e >= observed]
    # Cover the whole tail beyond the observed line in light red.
    ax.axvspan(observed, 0.95, color="#fbe3df", alpha=0.55, zorder=1)

    # Observed vertical line (brand red).
    ax.axvline(observed, color="#c0392b", linewidth=3.2, zorder=5)

    # "0 of 10,000 permutations reached this value" label in the tail.
    ax.text(
        observed + 0.008,
        y_hist_max * 0.82,
        f"0 of {n_perms:,} permutations\nreached this value",
        fontsize=10.5, color="#c0392b", va="top", ha="left",
        fontname="Georgia", style="italic",
    )

    # Numeric label on the observed line.
    ax.text(
        observed,
        y_hist_max * 1.07,
        f"observed match mean\n{observed:.3f}",
        fontsize=11, color="#c0392b", ha="center", va="bottom",
        fontweight="bold", fontname="Georgia",
    )

    # Tick-mark strips for echoes (blue) and legit criticism (green) just above 0.
    tick_y_legit = -y_hist_max * 0.055
    tick_y_echo = -y_hist_max * 0.12
    ax.plot(legit_scores, np.full_like(legit_scores, tick_y_legit),
            "|", color="#27ae60", markersize=14, markeredgewidth=2.0, zorder=4)
    ax.plot(echo_scores, np.full_like(echo_scores, tick_y_echo),
            "|", color="#3498db", markersize=14, markeredgewidth=2.0, zorder=4)

    # Labels for the tick strips.
    ax.text(0.252, tick_y_legit, "legit policy criticism  (n=8)",
            fontsize=9, color="#27ae60", va="center", ha="left", fontweight="bold")
    ax.text(0.252, tick_y_echo, "known echoes  (n=10)",
            fontsize=9, color="#3498db", va="center", ha="left", fontweight="bold")

    # Annotate null distribution mean with a faint dotted vertical.
    ax.axvline(null_mean, color="#666", linewidth=1.0, linestyle=":", alpha=0.8, zorder=3)
    ax.text(null_mean, y_hist_max * 0.96, f"null mean\n{null_mean:.3f}",
            fontsize=9, color="#555", ha="center", va="top")

    # Axes cosmetics.
    ax.set_xlim(0.25, 0.95)
    ax.set_ylim(-y_hist_max * 0.18, y_hist_max * 1.22)
    ax.set_xlabel("Cosine similarity (BGE-large-en-v1.5)", fontsize=11)
    ax.set_ylabel("Random pairs per bin  (n = 1,000)", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    # Hide negative y-axis tick labels so the tick strips don't clutter.
    yticks = [t for t in ax.get_yticks() if t >= 0]
    ax.set_yticks(yticks)

    # Title and subtitle.
    ax.set_title(
        "Observed match mean lies far beyond the random-pair null distribution",
        fontsize=13.5, fontweight="bold", loc="left", pad=32, fontname="Georgia",
    )
    subtitle = (
        f"Cohen's d = {cohens_d:.2f}   ·   AUC = {auc_mvr:.3f}   ·   "
        f"permutation p < 1e-4  ({n_perms:,} permutations)"
    )
    ax.text(0.0, 1.022, subtitle, transform=ax.transAxes,
            fontsize=10.5, color="#444", fontname="Georgia")

    # Caption.
    caption = (
        "Null built by drawing 1,000 random Soviet-modern passage pairs and scoring their cosine similarity. "
        "Permutation test shuffles match labels across 10,000 iterations; none produced a mean reaching the observed value."
    )
    fig.text(0.06, 0.01, caption, fontsize=8.5, color="#666",
             fontname="Georgia", style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = "viz_examples/permutation_null.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
