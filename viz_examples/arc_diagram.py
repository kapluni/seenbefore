"""
Decade-gap arc diagram: Soviet passages on a 1965–1990 axis, modern passages on
a 2015–2026 axis, with an arc for each of the 24 curated matches. Arc color
encodes shared trope; arc opacity encodes ensemble score.

A visual break between the two eras makes the gap explicit so readers do not
infer continuous transmission (which the data cannot show).

Outputs viz_examples/arc_diagram.png.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

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

# Two era axes mapped into a single x coordinate with a visual gap.
SOV_RANGE = (1965, 1990)
MOD_RANGE = (2015, 2026)
GAP_WIDTH = 4.0     # blank units between eras
ERA_WIDTH = 20.0    # total drawing width per era


def sov_x(year):
    frac = (year - SOV_RANGE[0]) / (SOV_RANGE[1] - SOV_RANGE[0])
    return frac * ERA_WIDTH


def mod_x(year):
    frac = (year - MOD_RANGE[0]) / (MOD_RANGE[1] - MOD_RANGE[0])
    return ERA_WIDTH + GAP_WIDTH + frac * ERA_WIDTH


def draw_arc(ax, x1, x2, color, alpha, linewidth):
    span = abs(x2 - x1)
    peak_y = 0.25 + 0.04 * span
    verts = [(x1, 0), (x1, peak_y), (x2, peak_y), (x2, 0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    patch = PathPatch(Path(verts, codes), fill=False, edgecolor=color,
                      alpha=alpha, linewidth=linewidth, capstyle="round")
    ax.add_patch(patch)


def main():
    with open("viz_data.json") as f:
        matches = json.load(f)["matches"]

    matches = [m for m in matches if m.get("sovietYear") and m.get("modernYear")]
    matches.sort(key=lambda m: -(m.get("ensembleScore") or m.get("similarity") or 0))

    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=150)
    ax.set_xlim(-1.2, 2 * ERA_WIDTH + GAP_WIDTH + 1.2)
    ax.set_ylim(-0.55, 1.6)

    # Era bands
    ax.axhspan(-0.05, 0.015, xmin=0.02, xmax=(ERA_WIDTH)/(2*ERA_WIDTH+GAP_WIDTH)+0.01,
               color="#fde8e4", zorder=1)
    ax.axhspan(-0.05, 0.015, xmin=(ERA_WIDTH+GAP_WIDTH)/(2*ERA_WIDTH+GAP_WIDTH)-0.01, xmax=0.98,
               color="#e5eef9", zorder=1)

    # "No data between" strip
    mid_l = ERA_WIDTH
    mid_r = ERA_WIDTH + GAP_WIDTH
    ax.fill_between([mid_l, mid_r], -0.05, 1.55, color="#f6f6f6", alpha=0.85, zorder=1)
    ax.text((mid_l+mid_r)/2, 1.48, "no data\nin this gap", ha="center", va="top",
            fontsize=8.5, color="#888", style="italic")

    # Tick marks and year labels per era
    sov_ticks = [1965, 1970, 1975, 1980, 1985, 1990]
    for y in sov_ticks:
        x = sov_x(y)
        ax.plot([x, x], [-0.035, -0.01], color="#666", linewidth=0.8)
        ax.text(x, -0.08, str(y), ha="center", va="top", fontsize=9, color="#333")
    mod_ticks = [2015, 2018, 2021, 2024]
    for y in mod_ticks:
        x = mod_x(y)
        ax.plot([x, x], [-0.035, -0.01], color="#666", linewidth=0.8)
        ax.text(x, -0.08, str(y), ha="center", va="top", fontsize=9, color="#333")

    # Era titles
    ax.text(ERA_WIDTH/2, -0.2, "Soviet propaganda corpus", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#c0392b", fontname="Georgia")
    ax.text(ERA_WIDTH + GAP_WIDTH + ERA_WIDTH/2, -0.2, "Modern antisemitic / anti-Zionist rhetoric",
            ha="center", va="top", fontsize=11, fontweight="bold", color="#2980b9", fontname="Georgia")

    # Arcs
    for m in matches:
        x1 = sov_x(m["sovietYear"])
        x2 = mod_x(m["modernYear"])
        tropes = m.get("tropes") or []
        color = TROPE_COLORS.get(tropes[0], "#555") if tropes else "#555"
        ens = m.get("ensembleScore") or m.get("similarity") or 0.5
        alpha = 0.35 + 0.55 * min(1.0, max(0.0, (ens - 0.55) / 0.35))
        lw = 1.2 + 2.4 * min(1.0, max(0.0, (ens - 0.55) / 0.35))
        draw_arc(ax, x1, x2, color, alpha, lw)

    # Point markers
    for m in matches:
        x1 = sov_x(m["sovietYear"])
        x2 = mod_x(m["modernYear"])
        ax.plot(x1, 0, "o", color="#c0392b", markersize=4, zorder=5)
        ax.plot(x2, 0, "o", color="#2980b9", markersize=4, zorder=5)

    # Legend (tropes)
    legend_tropes = sorted({(m.get("tropes") or [None])[0] for m in matches})
    legend_tropes = [t for t in legend_tropes if t]
    lx0 = 0
    ly = 1.35
    for t in legend_tropes:
        ax.add_patch(plt.Rectangle((lx0, ly-0.02), 0.35, 0.04, color=TROPE_COLORS[t]))
        ax.text(lx0 + 0.55, ly, TROPE_SHORT.get(t, t), fontsize=8.5, va="center", color="#333")
        lx0 += len(TROPE_SHORT.get(t, t)) * 0.18 + 2.0

    ax.set_title(f"Decade-gap arc diagram  ·  n={len(matches)} curated matches",
                 fontsize=13, fontweight="bold", loc="left", pad=12, fontname="Georgia")
    ax.text(0, 1.58,
            "Arc color = shared trope. Opacity and line width scale with ensemble score (0.55–0.90).",
            fontsize=9.5, color="#555", fontname="Georgia")

    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = "viz_examples/arc_diagram.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
