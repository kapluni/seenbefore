"""
Dumbbell plot: one row per curated match. Each row has a red dot at the
Soviet passage year and a blue dot at the modern passage year, joined by
a line colored by the match's primary trope. Rows are grouped by trope so
each rhetorical family forms a vertical bundle; within a trope rows are
sorted by ensemble score descending.

Era bands (pale red 1965-1990, pale blue 2015-2028) anchor the x-axis.
Line opacity and width scale with ensembleScore.

Outputs viz_examples/dumbbell.png.
"""
import os, json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))

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
TROPE_SHORT = {
    "ZIONISM_RACISM":          "Zionism = Racism",
    "ZIONISM_NAZISM":          "Zionism = Nazism",
    "ZIONISM_IMPERIALISM":     "Zionism = Imperialism",
    "JEWISH_CONSPIRACY":       "Jewish Conspiracy",
    "DELEGITIMIZATION":        "Delegitimization",
    "WEAPONIZED_ANTISEMITISM": "Weaponized Antisemitism",
    "DUAL_LOYALTY":            "Dual Loyalty",
    "BLOOD_LIBEL":             "Blood Libel",
    "ANTI_ZIONISM_PROGRESSIVE":"Progressive Anti-Zionism",
}

# Trope display order: by count desc (based on current corpus) then alpha.
TROPE_ORDER = [
    "ZIONISM_RACISM",
    "ZIONISM_NAZISM",
    "ZIONISM_IMPERIALISM",
    "BLOOD_LIBEL",
    "ANTI_ZIONISM_PROGRESSIVE",
    "WEAPONIZED_ANTISEMITISM",
    "JEWISH_CONSPIRACY",
    "DELEGITIMIZATION",
    "DUAL_LOYALTY",
]


def scale_opacity(ens, lo=0.55, hi=0.85):
    f = (ens - lo) / (hi - lo)
    f = max(0.0, min(1.0, f))
    return 0.35 + 0.55 * f  # 0.35..0.90


def scale_width(ens, lo=0.55, hi=0.85):
    f = (ens - lo) / (hi - lo)
    f = max(0.0, min(1.0, f))
    return 1.6 + 3.4 * f  # 1.6..5.0


def main():
    with open("viz_data.json") as f:
        matches = json.load(f)["matches"]

    matches = [m for m in matches if m.get("sovietYear") and m.get("modernYear") and m.get("tropes")]

    # Group by primary trope, then sort each group by ensembleScore desc.
    groups = {}
    for m in matches:
        t = m["tropes"][0]
        groups.setdefault(t, []).append(m)
    for t in groups:
        groups[t].sort(key=lambda x: -(x.get("ensembleScore") or x.get("similarity") or 0))

    ordered_tropes = [t for t in TROPE_ORDER if t in groups]
    rows = []   # list of (match, trope, is_first_in_group)
    for t in ordered_tropes:
        for i, m in enumerate(groups[t]):
            rows.append((m, t, i == 0))

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=160)

    # Y positions: top to bottom so first match is at top.
    y_positions = list(range(n_rows))[::-1]

    # X range.
    x_min, x_max = 1960, 2030

    # Era bands at bottom of the plot.
    band_y = -2.2
    band_h = 0.75
    ax.add_patch(Rectangle((1965, band_y), 1990 - 1965, band_h,
                           color="#fde8e4", zorder=1, linewidth=0))
    ax.add_patch(Rectangle((2015, band_y), 2028 - 2015, band_h,
                           color="#e5eef9", zorder=1, linewidth=0))
    ax.text((1965 + 1990) / 2, band_y + band_h / 2, "Soviet era  (1965–1990)",
            ha="center", va="center", fontsize=9.5, color="#c0392b",
            fontweight="bold", fontname="Georgia", zorder=2)
    ax.text((2015 + 2028) / 2, band_y + band_h / 2, "Modern era  (2015–2028)",
            ha="center", va="center", fontsize=9.5, color="#2980b9",
            fontweight="bold", fontname="Georgia", zorder=2)

    # Light horizontal grid at each row.
    for y in y_positions:
        ax.plot([x_min, x_max], [y, y], color="#f2f2f2", linewidth=0.7, zorder=1)

    # Dividers between trope groups.
    cursor = 0
    for t in ordered_tropes:
        cursor += len(groups[t])
        # divider line just below the last row of the group
        div_y = n_rows - cursor - 0.5
        if cursor < n_rows:
            ax.plot([x_min, x_max], [div_y, div_y],
                    color="#d0d0d0", linewidth=0.8, linestyle="-", zorder=1)

    # Dumbbells.
    for (m, t, is_first), y in zip(rows, y_positions):
        color = TROPE_COLORS[t]
        ens = m.get("ensembleScore") or m.get("similarity") or 0.5
        alpha = scale_opacity(ens)
        lw = scale_width(ens)
        sx = m["sovietYear"]
        mx = m["modernYear"]
        ax.plot([sx, mx], [y, y], color=color, alpha=alpha,
                linewidth=lw, solid_capstyle="round", zorder=3)
        ax.plot(sx, y, "o", color="#c0392b", markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.plot(mx, y, "o", color="#2980b9", markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    # Left-side trope swatches (one per row) and group labels on the first row.
    swatch_x = 1955  # just off the x-axis range
    for (m, t, is_first), y in zip(rows, y_positions):
        color = TROPE_COLORS[t]
        ax.add_patch(Rectangle((swatch_x, y - 0.28), 2.6, 0.56,
                               color=color, zorder=4, linewidth=0))
    # Group labels positioned on the first row of each group.
    cursor = 0
    for t in ordered_tropes:
        first_row_idx = cursor
        last_row_idx = cursor + len(groups[t]) - 1
        cursor += len(groups[t])
        # y of first row
        y_top = n_rows - first_row_idx - 1
        y_bot = n_rows - last_row_idx - 1
        y_mid = (y_top + y_bot) / 2
        ax.text(1953, y_mid, TROPE_SHORT[t],
                ha="right", va="center", fontsize=9.5, color=TROPE_COLORS[t],
                fontweight="bold", fontname="Georgia")
        # group count badge
        ax.text(1953, y_mid - 0.55, f"n={len(groups[t])}",
                ha="right", va="center", fontsize=8, color="#888")

    # Right-side match-id labels.
    for (m, t, is_first), y in zip(rows, y_positions):
        ax.text(2031, y, f"#{m['id']}", fontsize=8.5, va="center",
                ha="left", color="#555", fontname="Georgia")

    # X-axis ticks.
    ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030])
    ax.set_xlim(1940, 2038)
    ax.set_ylim(band_y - 0.5, n_rows - 0.3)

    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.tick_params(axis="x", labelsize=10, colors="#333")
    ax.set_xlabel("Year of passage", fontsize=11, color="#333", labelpad=8)

    # Legend strip on top: one swatch per trope present.
    legend_y = n_rows + 1.0
    ax.set_ylim(band_y - 0.5, legend_y + 1.6)
    lx = 1960
    ax.text(1960, legend_y + 1.15, "Trope (line color)",
            fontsize=9, color="#555", fontweight="bold", va="bottom")
    for t in ordered_tropes:
        label = TROPE_SHORT[t]
        ax.add_patch(Rectangle((lx, legend_y), 1.8, 0.45,
                               color=TROPE_COLORS[t], linewidth=0))
        ax.text(lx + 2.2, legend_y + 0.22, label,
                fontsize=8.8, va="center", color="#333")
        lx += 2.2 + len(label) * 0.62 + 2.0

    # Dot legend (Soviet / Modern) below trope legend.
    dot_y = legend_y - 0.9
    ax.plot(1960, dot_y, "o", color="#c0392b", markersize=6.5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(1961.6, dot_y, "Soviet passage year",
            fontsize=8.8, va="center", color="#333")
    ax.plot(1975, dot_y, "o", color="#2980b9", markersize=6.5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(1976.6, dot_y, "Modern passage year",
            fontsize=8.8, va="center", color="#333")
    ax.text(1994, dot_y, "line opacity/width ∝ ensemble score (0.55–0.85)",
            fontsize=8.5, va="center", color="#666", style="italic")

    # Title.
    ax.set_title(
        f"Soviet-to-modern match pairs, grouped by trope  ·  n={n_rows} curated matches",
        fontsize=13.5, fontweight="bold", loc="left", pad=14, fontname="Georgia",
    )
    ax.text(0.0, 1.01,
            "Each row is one match: Soviet year (red dot) joined to modern year (blue dot) by a trope-colored line.",
            transform=ax.transAxes,
            fontsize=10, color="#555", fontname="Georgia")

    fig.tight_layout()
    out = "viz_examples/dumbbell.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
