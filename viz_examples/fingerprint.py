"""
Rhetorical-fingerprint bars: one horizontal stacked bar per source, normalized
to 100%. Each bar's color mix = distribution of trope labels across all matches
involving that source. Soviet sources on top, modern datasets on bottom, with a
divider between eras.

The visual punch: a 1970 Moscow pamphlet should show a nearly identical color
mix as a 2020s Twitter dataset.

Outputs viz_examples/fingerprint.png.
"""
import os, json
from collections import Counter, defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

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


def main():
    with open("viz_data.json") as f:
        matches = json.load(f)["matches"]

    # Aggregate trope counts per Soviet source and per modern dataset.
    sov_counts = defaultdict(Counter)   # source -> Counter(trope)
    mod_counts = defaultdict(Counter)
    sov_n = Counter()
    mod_n = Counter()
    sov_year = {}
    mod_year_min = {}
    mod_year_max = {}

    for m in matches:
        ss = m["sovietSource"]
        ms = m["modernSource"]
        sov_n[ss] += 1
        mod_n[ms] += 1
        sov_year[ss] = m.get("sovietYear")
        if ms not in mod_year_min or (m.get("modernYear") and m["modernYear"] < mod_year_min[ms]):
            mod_year_min[ms] = m.get("modernYear")
        if ms not in mod_year_max or (m.get("modernYear") and m["modernYear"] > mod_year_max[ms]):
            mod_year_max[ms] = m.get("modernYear")
        for t in (m.get("tropes") or []):
            if t in TROPE_COLORS:
                sov_counts[ss][t] += 1
                mod_counts[ms][t] += 1

    # Sort Soviet sources by year ascending (oldest on top)
    soviet_rows = sorted(sov_counts.keys(), key=lambda s: (sov_year.get(s, 9999), s))
    # Sort modern datasets by match count descending (so largest on top of modern block)
    modern_rows = sorted(mod_counts.keys(), key=lambda s: (-mod_n[s], s))

    # Build bar layout: Soviet rows, a divider, then modern rows (top-to-bottom in y)
    # Matplotlib y increases upward, so we place higher rows at larger y values.
    rows = []  # (label, year_str, counts_dict, n, kind)
    for s in soviet_rows:
        rows.append((s, f"{sov_year.get(s, '')}", sov_counts[s], sov_n[s], "soviet"))
    rows.append(("__DIVIDER__", "", None, 0, "divider"))
    for s in modern_rows:
        yr_min = mod_year_min.get(s)
        yr_max = mod_year_max.get(s)
        if yr_min and yr_max and yr_min != yr_max:
            year_str = f"{yr_min}-{yr_max}"
        elif yr_min:
            year_str = f"{yr_min}"
        else:
            year_str = ""
        rows.append((s, year_str, mod_counts[s], mod_n[s], "modern"))

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("white")

    bar_h = 0.62
    # y-positions: top row at highest y
    ys = list(range(n_rows - 1, -1, -1))

    max_x = 1.0
    for idx, (label, year_str, counts, n, kind) in enumerate(rows):
        y = ys[idx]
        if kind == "divider":
            ax.axhline(y=y, color="#bbb", linestyle="--", linewidth=1.0, zorder=1)
            ax.text(0.5, y, "— Soviet  /  Modern divide —", ha="center", va="center",
                    fontsize=9.5, color="#888", fontname="Georgia",
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
            continue
        total = sum(counts.values())
        if total == 0:
            continue
        x_cursor = 0.0
        for trope in TROPE_ORDER:
            c = counts.get(trope, 0)
            if c == 0:
                continue
            w = c / total
            ax.barh(y, w, left=x_cursor, height=bar_h,
                    color=TROPE_COLORS[trope], edgecolor="white", linewidth=0.6,
                    zorder=2)
            # Only label segments that are wide enough
            if w >= 0.12:
                ax.text(x_cursor + w / 2, y, str(c),
                        ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold", fontname="Georgia")
            x_cursor += w

    # Left-side labels (source name + gray year) and right-side n counts
    for idx, (label, year_str, counts, n, kind) in enumerate(rows):
        y = ys[idx]
        if kind == "divider":
            continue
        color_accent = "#c0392b" if kind == "soviet" else "#2980b9"
        ax.text(-0.015, y, label, ha="right", va="center",
                fontsize=10.5, fontname="Georgia", color="#222")
        ax.text(-0.015, y - 0.28, year_str, ha="right", va="center",
                fontsize=8.5, fontname="Georgia", color="#888",
                style="italic")
        # Era dot
        ax.plot(-0.008, y, "o", markersize=6, color=color_accent,
                markeredgecolor="white", zorder=5, clip_on=False)
        ax.text(1.015, y, f"(n={n})", ha="left", va="center",
                fontsize=9.5, fontname="Georgia", color="#555")

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.4, n_rows - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=8.5, fontname="Georgia", color="#666")
    ax.set_yticks([])
    ax.set_xlabel("Share of trope occurrences (normalized per source)",
                  fontsize=9.5, fontname="Georgia", color="#333", labelpad=8)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="x", colors="#666", length=3)

    # Title via fig.text to control placement precisely
    fig.text(
        0.02, 0.96,
        "Rhetorical fingerprints  \u00b7  trope mix per source across 24 curated matches",
        fontsize=13.5, fontweight="bold", fontname="Georgia", color="#222",
    )
    fig.text(
        0.02, 0.925,
        "A 1970s Moscow pamphlet and a 2020s tweet dataset can carry near-identical trope signatures.",
        fontsize=10, color="#555", fontname="Georgia",
    )

    # Legend at the bottom using matplotlib's legend (handles spacing cleanly)
    present = [t for t in TROPE_ORDER
               if any(counts.get(t, 0) > 0 for _, _, counts, _, kind in rows if kind != "divider" and counts)]
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=TROPE_COLORS[t], edgecolor="white",
                     label=TROPE_SHORT[t]) for t in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(present), 5), frameon=False, fontsize=8.8,
              handlelength=1.2, handleheight=1.0, columnspacing=1.6,
              prop={"family": "Georgia", "size": 8.8})

    plt.subplots_adjust(left=0.23, right=0.93, top=0.88, bottom=0.18)

    out = "viz_examples/fingerprint.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    size_kb = os.path.getsize(out) / 1024
    print(f"Wrote {out}  ({size_kb:.1f} KB)  "
          f"soviet_sources={len(soviet_rows)} modern_sources={len(modern_rows)}")


if __name__ == "__main__":
    main()
