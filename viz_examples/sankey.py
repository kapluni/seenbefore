"""
Sankey diagram: Soviet trope -> Modern trope, from real viz_data.json matches.
Re-classifies each match's modern text using the keyword trope taxonomy, then
draws a Plotly Sankey connecting the Soviet trope side of each match to every
modern-side trope detected in the same match.

Outputs viz_examples/sankey.html (interactive, hover shows match count).
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import plotly.graph_objects as go
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
    "WEAPONIZED_ANTISEMITISM": "Weaponized Antisemitism",
    "DUAL_LOYALTY":            "Dual Loyalty",
    "BLOOD_LIBEL":             "Blood Libel",
    "ANTI_ZIONISM_PROGRESSIVE":"Progressive Anti-Zionism",
}


def hex_to_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}, {a})"


def classify(text, processor):
    return processor.classify_tropes_keyword(Passage(
        id="tmp", text=text, source="", source_title="", author="", year=0, language="en", corpus="x"
    ))


def main():
    with open("viz_data.json") as f:
        matches = json.load(f)["matches"]

    processor = CorpusProcessor()

    flows = {}
    for m in matches:
        sov_text = m.get("sovietTextFull") or m.get("sovietText", "")
        mod_text = m.get("modernTextFull") or m.get("modernText", "")
        sov_tropes = classify(sov_text, processor) or m.get("tropes", [])
        mod_tropes = classify(mod_text, processor) or m.get("tropes", [])
        for s in sov_tropes:
            for mt in mod_tropes:
                flows[(s, mt)] = flows.get((s, mt), 0) + 1

    soviet_present = sorted({s for s, _ in flows}, key=lambda x: TROPE_ORDER.index(x))
    modern_present = sorted({mm for _, mm in flows}, key=lambda x: TROPE_ORDER.index(x))

    node_labels = [f"Soviet: {TROPE_SHORT[t]}" for t in soviet_present] + \
                  [f"Modern: {TROPE_SHORT[t]}" for t in modern_present]
    node_colors = [TROPE_COLORS[t] for t in soviet_present] + \
                  [TROPE_COLORS[t] for t in modern_present]

    sov_idx = {t: i for i, t in enumerate(soviet_present)}
    mod_idx = {t: i + len(soviet_present) for i, t in enumerate(modern_present)}

    sources, targets, values, link_colors = [], [], [], []
    for (s, mt), v in sorted(flows.items(), key=lambda x: -x[1]):
        sources.append(sov_idx[s])
        targets.append(mod_idx[mt])
        values.append(v)
        link_colors.append(hex_to_rgba(TROPE_COLORS[s], 0.45))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=18, line=dict(color="#333", width=0.5),
            label=node_labels, color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors,
                  hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>Matches: %{value}<extra></extra>"),
    )])
    fig.update_layout(
        title=dict(
            text=(f"Trope Flow: Soviet → Modern  ·  n={len(matches)} matches<br>"
                  f"<span style='font-size:12px;color:#666'>"
                  "Re-classified per match; shows rhetorical-category correspondence "
                  "(not causal transmission).</span>"),
            x=0.02, xanchor="left",
        ),
        font=dict(family="Georgia, serif", size=13),
        paper_bgcolor="white", plot_bgcolor="white",
        width=1100, height=720, margin=dict(l=40, r=40, t=90, b=40),
    )
    out = "viz_examples/sankey.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    print(f"Flow edges: {len(flows)}  total weighted: {sum(flows.values())}")


if __name__ == "__main__":
    main()
