"""
9x9 Soviet->Modern trope heatmap over the 24 curated matches.

Each match's Soviet claim (Claude-extracted, dense ~15 words) is classified with
the keyword trope taxonomy; same for the modern claim. Cell (i,j) counts matches
whose Soviet side carries trope i and whose modern side carries trope j. Two
subplots: left = multi-label (count every detected trope on each side),
right = top-1 (canonical Soviet tropes[0] vs. the first detected modern trope).

The diagonal should dominate if Soviet and modern rhetoric share structure.

Outputs viz_examples/heatmap.png.
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from embedding_pipeline import TROPE_TAXONOMY, CorpusProcessor, Passage

TROPE_COLORS = {
    "ZIONISM_RACISM":          "#c0392b",
    "ZIONISM_NAZISM":          "#8e44ad",
    "ZIONISM_IMPERIALISM":     "#2980b9",
    "JEWISH_CONSPIRACY":       "#d35400",
    "DELEGITIMIZATION":        "#27ae60",
    "WEAPONIZED_ANTISEMITISM": "#f39c12",
    "DUAL_LOYALTY":            "#1abc9c",
    "BLOOD_LIBEL":             "#e74c3c",
    "ANTI_ZIONISM_PROGRESSIVE":"#3498db",
}
TROPE_ORDER = list(TROPE_COLORS.keys())
TROPE_SHORT = {
    "ZIONISM_RACISM":          "Zionism = Racism",
    "ZIONISM_NAZISM":          "Zionism = Nazism",
    "ZIONISM_IMPERIALISM":     "Zionism = Imperialism",
    "JEWISH_CONSPIRACY":       "Jewish Conspiracy",
    "DELEGITIMIZATION":        "Delegitimization",
    "WEAPONIZED_ANTISEMITISM": "Weaponized Antisem.",
    "DUAL_LOYALTY":            "Dual Loyalty",
    "BLOOD_LIBEL":             "Blood Libel",
    "ANTI_ZIONISM_PROGRESSIVE":"Progressive AZ",
}


def classify(text, processor):
    if not text:
        return []
    return processor.classify_tropes_keyword(Passage(
        id="tmp", text=text, source="", source_title="", author="", year=0,
        language="en", corpus="x"
    ))


def build_matrices(matches, processor):
    n = len(TROPE_ORDER)
    idx = {t: i for i, t in enumerate(TROPE_ORDER)}
    multi = np.zeros((n, n), dtype=int)
    top1 = np.zeros((n, n), dtype=int)

    for m in matches:
        claim_pair = m.get("claimPair") or {}
        sov_claim = claim_pair.get("sovietClaim", "") or ""
        mod_claim = claim_pair.get("modernClaim", "") or ""

        # Soviet classification: claim first, fall back to match's tropes list,
        # then fall back to the full passage.
        sov_tropes = classify(sov_claim, processor)
        if not sov_tropes:
            sov_tropes = [t for t in (m.get("tropes") or []) if t in idx]
        if not sov_tropes:
            sov_tropes = classify(m.get("sovietTextFull") or m.get("sovietText", ""), processor)

        mod_tropes = classify(mod_claim, processor)
        if not mod_tropes:
            mod_tropes = classify(m.get("modernTextFull") or m.get("modernText", ""), processor)

        # Multi-label: every (i,j) combination
        for s in sov_tropes:
            for mt in mod_tropes:
                if s in idx and mt in idx:
                    multi[idx[s], idx[mt]] += 1

        # Top-1: canonical Soviet tropes[0] (matches the curated label),
        # dominant modern = first detected from claim (falls back to passage)
        canonical_s = (m.get("tropes") or [None])[0]
        if canonical_s is None and sov_tropes:
            canonical_s = sov_tropes[0]
        canonical_m = mod_tropes[0] if mod_tropes else None
        if canonical_s in idx and canonical_m in idx:
            top1[idx[canonical_s], idx[canonical_m]] += 1

    return multi, top1


def draw_heatmap(ax, matrix, title):
    n = matrix.shape[0]
    labels = [TROPE_SHORT[t] for t in TROPE_ORDER]

    # Base colormap; deepen diagonal cells by overlaying a saturation bump.
    base_cmap = plt.get_cmap("Reds")
    vmax = max(matrix.max(), 1)
    im = ax.imshow(matrix, cmap=base_cmap, vmin=0, vmax=vmax, aspect="equal")

    # Diagonal highlight: draw a thin darker border on diagonal non-zero cells.
    for i in range(n):
        v = matrix[i, i]
        if v > 0:
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                 edgecolor="#7a1a0a", linewidth=1.8, zorder=3)
            ax.add_patch(rect)

    # Annotate non-zero cells
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if v > 0:
                # Text color: white on dark cells, dark on light
                color = "white" if v / vmax > 0.55 else "#333"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=9.5, color=color, fontweight=weight,
                        fontname="Georgia")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5,
                       fontname="Georgia", color="#333")
    ax.set_yticklabels(labels, fontsize=8.5, fontname="Georgia", color="#333")
    ax.set_xlabel("Modern-side trope", fontsize=10, fontname="Georgia",
                  color="#2980b9", labelpad=8)
    ax.set_ylabel("Soviet-side trope", fontsize=10, fontname="Georgia",
                  color="#c0392b", labelpad=8)

    # Marginal totals: right side = row totals, bottom = column totals
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    for i, v in enumerate(row_totals):
        ax.text(n - 0.3, i, f" {v}", va="center", ha="left",
                fontsize=9, color="#c0392b", fontweight="bold",
                fontname="Georgia", transform=ax.transData)
    for j, v in enumerate(col_totals):
        ax.text(j, n - 0.3, str(v), va="top", ha="center",
                fontsize=9, color="#2980b9", fontweight="bold",
                fontname="Georgia", transform=ax.transData)

    # Keep marginal totals visible: extend x/y limits slightly
    ax.set_xlim(-0.6, n + 0.4)
    ax.set_ylim(n + 0.4, -0.6)  # inverted because imshow

    ax.set_title(title, fontsize=11.5, fontname="Georgia",
                 fontweight="bold", pad=10, loc="left")

    # Clean spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#ccc")

    return im


def main():
    with open("viz_data.json") as f:
        matches = json.load(f)["matches"]

    processor = CorpusProcessor()
    multi, top1 = build_matrices(matches, processor)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), dpi=160,
                             gridspec_kw={"wspace": 0.55})
    fig.patch.set_facecolor("white")

    im1 = draw_heatmap(axes[0], multi,
                       f"Multi-label  ·  every detected trope counted  ·  total={int(multi.sum())}")
    im2 = draw_heatmap(axes[1], top1,
                       f"Top-1  ·  canonical Soviet label × dominant modern  ·  total={int(top1.sum())}")

    # Shared colorbar on the right edge
    cbar = fig.colorbar(im1, ax=axes.tolist(), shrink=0.62, pad=0.02,
                        fraction=0.025)
    cbar.set_label("Matches in cell", fontsize=9, fontname="Georgia", color="#333")
    cbar.ax.tick_params(labelsize=8)

    # Diagonal dominance indicator
    def diag_share(mat):
        tot = mat.sum()
        return (np.trace(mat) / tot) if tot else 0.0

    d_multi = diag_share(multi)
    d_top1 = diag_share(top1)

    fig.suptitle(
        f"Shared trope structure across {len(matches)} matches  ·  Diagonal = same trope both sides",
        fontsize=14, fontweight="bold", fontname="Georgia", x=0.02, ha="left", y=0.995,
    )
    fig.text(
        0.02, 0.955,
        "Rows = Soviet trope (from Claude-extracted claim · keyword taxonomy); "
        "columns = Modern trope (same). "
        f"Diagonal share: multi-label {d_multi:.0%}, top-1 {d_top1:.0%}.",
        fontsize=9.5, color="#555", fontname="Georgia",
    )

    out = "viz_examples/heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    size_kb = os.path.getsize(out) / 1024
    print(f"Wrote {out}  ({size_kb:.1f} KB)  "
          f"multi total={int(multi.sum())} diag={d_multi:.2f}  "
          f"top1 total={int(top1.sum())} diag={d_top1:.2f}")


if __name__ == "__main__":
    main()
