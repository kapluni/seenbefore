import { useState, useEffect, useRef, useMemo } from "react";

// ============================================================
// VizPrototypes — 5 visualization concepts using real match data
// Access via /?prototypes
// ============================================================

const TROPE_COLORS = {
  ZIONISM_RACISM: "#c0392b", ZIONISM_NAZISM: "#8e44ad", ZIONISM_IMPERIALISM: "#2980b9",
  JEWISH_CONSPIRACY: "#d35400", DELEGITIMIZATION: "#27ae60", WEAPONIZED_ANTISEMITISM: "#f39c12",
  DUAL_LOYALTY: "#1abc9c", BLOOD_LIBEL: "#e74c3c", ANTI_ZIONISM_PROGRESSIVE: "#3498db",
};

const TROPE_SHORT = {
  ZIONISM_RACISM: "Racism", ZIONISM_NAZISM: "Nazism", ZIONISM_IMPERIALISM: "Imperialism",
  JEWISH_CONSPIRACY: "Conspiracy", DELEGITIMIZATION: "Delegitimize",
  WEAPONIZED_ANTISEMITISM: "Weaponized AS", DUAL_LOYALTY: "Dual Loyalty",
  BLOOD_LIBEL: "Blood Libel", ANTI_ZIONISM_PROGRESSIVE: "Progressive",
};

const STOP_WORDS = new Set([
  "the","a","an","and","or","but","in","on","at","to","for","of","with","by",
  "from","is","are","was","were","be","been","being","have","has","had","do",
  "does","did","will","would","could","should","may","might","shall","can",
  "need","must","it","its","this","that","these","those","he","she","they",
  "we","you","i","me","my","your","his","her","our","their","them","us","him",
  "who","whom","which","what","where","when","how","why","if","then","than",
  "so","as","not","no","nor","up","out","off","all","each","every","both",
  "few","more","most","other","some","such","only","own","same","also","very",
  "just","about","into","over","after","before","between","under","again",
  "there","here","once","during","while","because","any","against","never",
]);

const TROPE_KEYWORDS = {
  ZIONISM_RACISM: ["racism","racist","apartheid","supremacy","supremacist","racial","segregation","discrimination","chauvini"],
  ZIONISM_NAZISM: ["nazi","nazism","fascis","hitler","genocide","holocaust","extermination","third reich"],
  ZIONISM_IMPERIALISM: ["imperial","colonial","colonialism","settler","occupation","neo-colonial","exploitat"],
  JEWISH_CONSPIRACY: ["control","lobby","manipulat","media","finance","power","influence","world domination","cabal","tentacle"],
  DELEGITIMIZATION: ["entity","illegitimate","no right","artificial state","invented"],
  WEAPONIZED_ANTISEMITISM: ["weaponiz","silence","critic","antisemit","card","smear","shield"],
  DUAL_LOYALTY: ["loyalty","allegiance","fifth column","agent","traitor","spy","foreign"],
  BLOOD_LIBEL: ["blood","child","kill","murder","slaughter","massacre","baby","deliberat"],
  ANTI_ZIONISM_PROGRESSIVE: ["liberation","progressive","solidarity","resist","struggle","duty","movement","unite","workers"],
};

// ============================================================
// DATA LOADING
// ============================================================
function useMatchData() {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch("/viz_data.json")
      .then(r => r.ok ? r.json() : null)
      .then(d => setData(d))
      .catch(() => setData(null));
  }, []);
  return data;
}

// ============================================================
// SHARED UTILITIES
// ============================================================

function getSharedWords(text1, text2) {
  const normalize = t => t.toLowerCase().replace(/[^a-z\s]/g, "").split(/\s+/).filter(w => w.length > 3 && !STOP_WORDS.has(w));
  const words1 = new Set(normalize(text1));
  const words2 = new Set(normalize(text2));
  return [...words1].filter(w => words2.has(w));
}

function highlightText(text, keywords, color) {
  if (!keywords.length) return text;
  const pattern = new RegExp(`\\b(${keywords.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join("|")})`, "gi");
  const parts = text.split(pattern);
  return parts.map((part, i) =>
    pattern.test(part)
      ? <mark key={i} style={{ background: color + "44", color: color, borderBottom: `2px solid ${color}`, padding: "0 2px", borderRadius: 2 }}>{part}</mark>
      : part
  );
}

function getTropeKeywordsForMatch(tropes) {
  const kw = [];
  for (const t of tropes) {
    if (TROPE_KEYWORDS[t]) kw.push(...TROPE_KEYWORDS[t]);
  }
  return [...new Set(kw)];
}

// ============================================================
// 1. HIGHLIGHTED TEXT DIFF
// ============================================================
function HighlightedDiff({ matches }) {
  const [idx, setIdx] = useState(0);
  const m = matches[idx];
  if (!m) return null;

  const shared = getSharedWords(m.sovietTextFull || m.sovietText, m.modernTextFull || m.modernText);
  const tropeKw = getTropeKeywordsForMatch(m.tropes || []);
  const allKeywords = [...new Set([...shared, ...tropeKw])];

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={() => setIdx(Math.max(0, idx - 1))} disabled={idx === 0} style={navBtn}>Prev</button>
        <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>Match {idx + 1} / {matches.length}</span>
        <button onClick={() => setIdx(Math.min(matches.length - 1, idx + 1))} disabled={idx === matches.length - 1} style={navBtn}>Next</button>
        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted)" }}>
          {allKeywords.length} shared terms highlighted
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div style={{ borderLeft: "3px solid #c0392b", paddingLeft: 16 }}>
          <div style={{ fontSize: 10, letterSpacing: 2, color: "#c0392b", fontWeight: 700, marginBottom: 8 }}>
            THEN — {m.sovietYear} • {m.sovietSource}
          </div>
          <div style={{ fontFamily: "Georgia, serif", fontSize: 14, lineHeight: 1.7, color: "var(--text-primary)" }}>
            "{highlightText(m.sovietTextFull || m.sovietText, allKeywords, "#e67e22")}"
          </div>
        </div>
        <div style={{ borderLeft: "3px solid #3498db", paddingLeft: 16 }}>
          <div style={{ fontSize: 10, letterSpacing: 2, color: "#3498db", fontWeight: 700, marginBottom: 8 }}>
            NOW — {m.modernYear} • {m.modernSource}
          </div>
          <div style={{ fontFamily: "Georgia, serif", fontSize: 14, lineHeight: 1.7, color: "var(--text-primary)" }}>
            "{highlightText(m.modernTextFull || m.modernText, allKeywords, "#e67e22")}"
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 6, marginTop: 12, flexWrap: "wrap" }}>
        {(m.tropes || []).map(t => (
          <span key={t} style={{
            background: (TROPE_COLORS[t] || "#888") + "22",
            border: `1px solid ${(TROPE_COLORS[t] || "#888")}55`,
            color: TROPE_COLORS[t] || "#888",
            padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600,
          }}>{TROPE_SHORT[t] || t}</span>
        ))}
        <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--text-muted)" }}>
          Similarity: <strong style={{ color: "#e67e22" }}>{Math.round((m.ensembleScore || m.similarity) * 100)}%</strong>
        </span>
      </div>

      {m.echo_explanation && (
        <div style={{ fontSize: 12, color: "var(--text-secondary)", fontStyle: "italic", marginTop: 12, padding: "8px 12px", background: "var(--bg-card-alt)", borderRadius: 6 }}>
          {m.echo_explanation}
        </div>
      )}

      {allKeywords.length > 0 && (
        <div style={{ marginTop: 12, padding: "8px 12px", background: "var(--bg-card-alt)", borderRadius: 6 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4, letterSpacing: 1 }}>SHARED VOCABULARY</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {allKeywords.slice(0, 20).map(w => (
              <span key={w} style={{ fontSize: 11, padding: "1px 6px", background: "#e67e2233", color: "#e67e22", borderRadius: 3 }}>{w}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// 2. SANKEY FLOW DIAGRAM (pure SVG)
// ============================================================
function SankeyFlow({ matches }) {
  const svgRef = useRef(null);
  const [hoveredLink, setHoveredLink] = useState(null);

  const layout = useMemo(() => {
    const sources = {};
    const tropes = {};
    const moderns = {};

    for (const m of matches) {
      const src = m.sovietSource || "Unknown";
      sources[src] = (sources[src] || 0) + 1;
      const mod = m.modernSource || "Unknown";
      moderns[mod] = (moderns[mod] || 0) + 1;
      for (const t of (m.tropes || [])) {
        tropes[t] = (tropes[t] || 0) + 1;
      }
    }

    const links = [];
    const srcTrope = {};
    const tropemod = {};

    for (const m of matches) {
      const src = m.sovietSource;
      const mod = m.modernSource;
      for (const t of (m.tropes || [])) {
        const k1 = `${src}|${t}`;
        srcTrope[k1] = (srcTrope[k1] || 0) + 1;
        const k2 = `${t}|${mod}`;
        tropemod[k2] = (tropemod[k2] || 0) + 1;
      }
    }

    for (const [k, v] of Object.entries(srcTrope)) {
      const [src, trope] = k.split("|");
      links.push({ from: src, to: trope, value: v, type: "src-trope" });
    }
    for (const [k, v] of Object.entries(tropemod)) {
      const [trope, mod] = k.split("|");
      links.push({ from: trope, to: mod, value: v, type: "trope-mod" });
    }

    const srcList = Object.entries(sources).sort((a, b) => b[1] - a[1]);
    const tropeList = Object.entries(tropes).sort((a, b) => b[1] - a[1]);
    const modList = Object.entries(moderns).sort((a, b) => b[1] - a[1]);

    return { srcList, tropeList, modList, links };
  }, [matches]);

  const W = 860, H = 420, colX = [0, 340, 680], nodeW = 16, pad = 8;

  const positionNodes = (list, x, totalH) => {
    const total = list.reduce((s, [, v]) => s + v, 0);
    let y = pad;
    return list.map(([name, value]) => {
      const h = Math.max(14, ((value / total) * (totalH - pad * (list.length + 1))));
      const node = { name, x, y, w: nodeW, h, value };
      y += h + pad;
      return node;
    });
  };

  const srcNodes = positionNodes(layout.srcList, colX[0], H);
  const tropeNodes = positionNodes(layout.tropeList, colX[1], H);
  const modNodes = positionNodes(layout.modList, colX[2], H);

  const nodeMap = {};
  for (const n of [...srcNodes, ...tropeNodes, ...modNodes]) nodeMap[n.name] = n;

  const svgLinks = layout.links.map((link, i) => {
    const fromNode = nodeMap[link.from];
    const toNode = nodeMap[link.to];
    if (!fromNode || !toNode) return null;
    const fromX = fromNode.x + fromNode.w;
    const fromY = fromNode.y + fromNode.h / 2;
    const toX = toNode.x;
    const toY = toNode.y + toNode.h / 2;
    const thickness = Math.max(2, link.value * 3);
    const midX = (fromX + toX) / 2;
    const color = link.type === "src-trope" ? (TROPE_COLORS[link.to] || "#888") : (TROPE_COLORS[link.from] || "#888");
    const isHovered = hoveredLink === i;
    return {
      key: i,
      path: `M${fromX},${fromY} C${midX},${fromY} ${midX},${toY} ${toX},${toY}`,
      thickness,
      color,
      opacity: hoveredLink === null ? 0.35 : isHovered ? 0.8 : 0.1,
      link,
    };
  }).filter(Boolean);

  const shortSource = (name) => {
    if (name.length <= 24) return name;
    if (name.includes("Instrument")) return "Instrument of Reaction";
    if (name.includes("Caution")) return "Caution: Zionism!";
    if (name.includes("Enemy")) return "Enemy of Peace";
    if (name.includes("Aims")) return "AZC: Aims & Tasks";
    if (name.includes("ISCA Twitter")) return "ISCA Twitter";
    if (name.includes("ClassData")) return "ISCA ClassData";
    if (name.includes("H.E.A.T")) return "ADL H.E.A.T. Map";
    if (name.includes("CONAN")) return "CONAN Dialogue";
    return name.slice(0, 22) + "…";
  };

  return (
    <div style={{ overflowX: "auto" }}>
      <svg ref={svgRef} viewBox={`-120 -10 ${W + 240} ${H + 20}`} width="100%" style={{ maxHeight: 460 }}>
        {/* Column labels */}
        <text x={colX[0] + nodeW / 2} y={-2} fill="var(--text-muted)" fontSize={10} textAnchor="middle" fontWeight={700}>SOVIET SOURCES</text>
        <text x={colX[1] + nodeW / 2} y={-2} fill="var(--text-muted)" fontSize={10} textAnchor="middle" fontWeight={700}>TROPES</text>
        <text x={colX[2] + nodeW / 2} y={-2} fill="var(--text-muted)" fontSize={10} textAnchor="middle" fontWeight={700}>MODERN SOURCES</text>

        {/* Links */}
        {svgLinks.map(sl => (
          <path
            key={sl.key}
            d={sl.path}
            fill="none"
            stroke={sl.color}
            strokeWidth={sl.thickness}
            opacity={sl.opacity}
            onMouseEnter={() => setHoveredLink(sl.key)}
            onMouseLeave={() => setHoveredLink(null)}
            style={{ cursor: "pointer", transition: "opacity 0.2s" }}
          />
        ))}

        {/* Source nodes */}
        {srcNodes.map(n => (
          <g key={n.name}>
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={3} fill="#c0392b" />
            <text x={n.x - 4} y={n.y + n.h / 2} fill="#c0392b" fontSize={10} textAnchor="end" dominantBaseline="middle">
              {shortSource(n.name)}
            </text>
          </g>
        ))}

        {/* Trope nodes */}
        {tropeNodes.map(n => (
          <g key={n.name}>
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={3} fill={TROPE_COLORS[n.name] || "#888"} />
            <text x={n.x + nodeW + 4} y={n.y + n.h / 2} fill={TROPE_COLORS[n.name] || "#888"} fontSize={10} dominantBaseline="middle">
              {TROPE_SHORT[n.name] || n.name}
            </text>
          </g>
        ))}

        {/* Modern nodes */}
        {modNodes.map(n => (
          <g key={n.name}>
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={3} fill="#3498db" />
            <text x={n.x + nodeW + 4} y={n.y + n.h / 2} fill="#3498db" fontSize={10} dominantBaseline="middle">
              {shortSource(n.name)}
            </text>
          </g>
        ))}
      </svg>

      {hoveredLink !== null && svgLinks[hoveredLink] && (
        <div style={{ textAlign: "center", fontSize: 11, color: "var(--text-secondary)", marginTop: 4 }}>
          <strong>{svgLinks[hoveredLink].link.from}</strong> → <strong>{svgLinks[hoveredLink].link.to}</strong> ({svgLinks[hoveredLink].link.value} matches)
        </div>
      )}
    </div>
  );
}

// ============================================================
// 3. NETWORK GRAPH (pure SVG bipartite)
// ============================================================
function NetworkGraph({ matches }) {
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoveredMatch, setHoveredMatch] = useState(null);

  const graph = useMemo(() => {
    const sovietNodes = {};
    const modernNodes = {};
    const edges = [];

    for (const m of matches) {
      const sKey = m.sovietSource + "|" + (m.sovietText || "").slice(0, 40);
      const mKey = m.modernSource + "|" + (m.modernText || "").slice(0, 40);

      if (!sovietNodes[sKey]) {
        sovietNodes[sKey] = { id: sKey, label: m.sovietSource, year: m.sovietYear, text: (m.sovietText || "").slice(0, 80), tropes: new Set(), count: 0 };
      }
      for (const t of (m.tropes || [])) sovietNodes[sKey].tropes.add(t);
      sovietNodes[sKey].count++;

      if (!modernNodes[mKey]) {
        modernNodes[mKey] = { id: mKey, label: m.modernSource, year: m.modernYear, text: (m.modernText || "").slice(0, 80), tropes: new Set(), count: 0 };
      }
      for (const t of (m.tropes || [])) modernNodes[mKey].tropes.add(t);
      modernNodes[mKey].count++;

      edges.push({
        from: sKey, to: mKey,
        similarity: m.ensembleScore || m.similarity,
        tropes: m.tropes || [],
        match: m,
      });
    }

    return {
      sovietNodes: Object.values(sovietNodes),
      modernNodes: Object.values(modernNodes),
      edges,
    };
  }, [matches]);

  const W = 860, H = Math.max(400, Math.max(graph.sovietNodes.length, graph.modernNodes.length) * 28 + 40);
  const leftX = 100, rightX = W - 100;

  const posY = (list, totalH) => {
    const gap = totalH / (list.length + 1);
    return list.map((n, i) => ({ ...n, x: 0, y: gap * (i + 1) }));
  };

  const sNodes = posY(graph.sovietNodes, H).map(n => ({ ...n, x: leftX }));
  const mNodes = posY(graph.modernNodes, H).map(n => ({ ...n, x: rightX }));

  const nodeById = {};
  for (const n of sNodes) nodeById[n.id] = n;
  for (const n of mNodes) nodeById[n.id] = n;

  const isConnected = (nodeId) => {
    if (!hoveredNode) return true;
    if (nodeId === hoveredNode) return true;
    return graph.edges.some(e => (e.from === hoveredNode && e.to === nodeId) || (e.to === hoveredNode && e.from === nodeId));
  };

  return (
    <div style={{ overflowX: "auto" }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxHeight: 500 }}>
        <text x={leftX} y={16} fill="#c0392b" fontSize={11} textAnchor="middle" fontWeight={700}>SOVIET</text>
        <text x={rightX} y={16} fill="#3498db" fontSize={11} textAnchor="middle" fontWeight={700}>MODERN</text>

        {graph.edges.map((e, i) => {
          const from = nodeById[e.from];
          const to = nodeById[e.to];
          if (!from || !to) return null;
          const trope = e.tropes[0];
          const color = TROPE_COLORS[trope] || "#555";
          const active = hoveredNode === null || hoveredNode === e.from || hoveredNode === e.to;
          const isHovMatch = hoveredMatch === i;
          return (
            <path
              key={i}
              d={`M${from.x + 8},${from.y} C${W / 2},${from.y} ${W / 2},${to.y} ${to.x - 8},${to.y}`}
              fill="none"
              stroke={color}
              strokeWidth={isHovMatch ? 3 : 1.5}
              opacity={active ? (isHovMatch ? 0.9 : 0.4) : 0.06}
              onMouseEnter={() => setHoveredMatch(i)}
              onMouseLeave={() => setHoveredMatch(null)}
              style={{ cursor: "pointer", transition: "opacity 0.2s" }}
            />
          );
        })}

        {sNodes.map(n => {
          const r = 5 + n.count * 2;
          const trope = [...(n.tropes || [])][0];
          const active = isConnected(n.id);
          return (
            <g key={n.id} opacity={active ? 1 : 0.2} style={{ cursor: "pointer", transition: "opacity 0.2s" }}
               onMouseEnter={() => setHoveredNode(n.id)} onMouseLeave={() => setHoveredNode(null)}>
              <circle cx={n.x} cy={n.y} r={r} fill="#c0392b" stroke={hoveredNode === n.id ? "#fff" : "#c0392b44"} strokeWidth={hoveredNode === n.id ? 2 : 1} />
              <text x={n.x - r - 6} y={n.y} fill="var(--text-secondary)" fontSize={9} textAnchor="end" dominantBaseline="middle">
                {n.text.slice(0, 35)}…
              </text>
            </g>
          );
        })}

        {mNodes.map(n => {
          const r = 5 + n.count * 2;
          const active = isConnected(n.id);
          return (
            <g key={n.id} opacity={active ? 1 : 0.2} style={{ cursor: "pointer", transition: "opacity 0.2s" }}
               onMouseEnter={() => setHoveredNode(n.id)} onMouseLeave={() => setHoveredNode(null)}>
              <circle cx={n.x} cy={n.y} r={r} fill="#3498db" stroke={hoveredNode === n.id ? "#fff" : "#3498db44"} strokeWidth={hoveredNode === n.id ? 2 : 1} />
              <text x={n.x + r + 6} y={n.y} fill="var(--text-secondary)" fontSize={9} dominantBaseline="middle">
                {n.text.slice(0, 35)}…
              </text>
            </g>
          );
        })}
      </svg>

      {hoveredMatch !== null && graph.edges[hoveredMatch] && (
        <div style={{ padding: "8px 12px", background: "var(--bg-card-alt)", borderRadius: 6, fontSize: 12, color: "var(--text-primary)", marginTop: 4 }}>
          <strong style={{ color: "#c0392b" }}>{graph.edges[hoveredMatch].match.sovietYear}:</strong> "{(graph.edges[hoveredMatch].match.sovietText || "").slice(0, 100)}…"
          <br />
          <strong style={{ color: "#3498db" }}>{graph.edges[hoveredMatch].match.modernYear}:</strong> "{(graph.edges[hoveredMatch].match.modernText || "").slice(0, 100)}…"
          <br />
          <span style={{ color: "var(--text-muted)" }}>Similarity: {Math.round((graph.edges[hoveredMatch].match.ensembleScore || graph.edges[hoveredMatch].match.similarity) * 100)}%</span>
        </div>
      )}
    </div>
  );
}

// ============================================================
// 4. PROPAGANDA PLAYBOOK (process diagram)
// ============================================================
const PLAYBOOK_STEPS = [
  {
    num: 1, trope: "DELEGITIMIZATION", title: "Deny Jewish Nationhood",
    soviet: "Deny that Jews are a nation. Portray Israel as an artificial entity.",
    modern: "\"Israel has no right to exist as an ethnostate.\"",
  },
  {
    num: 2, trope: "ZIONISM_RACISM", title: "Label Zionism as Racism",
    soviet: "Zionism is racism, chauvinism, and racial intolerance.",
    modern: "\"Zionism is a racist ideology. Israel is an apartheid state.\"",
  },
  {
    num: 3, trope: "ZIONISM_NAZISM", title: "Equate Zionists with Nazis",
    soviet: "Draw equation: Zionism = Fascism = Hitlerism.",
    modern: "\"Zionists are basically Nazis. Gaza is a concentration camp.\"",
  },
  {
    num: 4, trope: "ZIONISM_IMPERIALISM", title: "Frame as Imperial Tool",
    soviet: "Zionism serves colonialism and fights liberation movements.",
    modern: "\"Israel is a settler-colonial project of Western imperialism.\"",
  },
  {
    num: 5, trope: "JEWISH_CONSPIRACY", title: "Invoke Zionist Control",
    soviet: "Zionist Concern controls finance, media, and intelligence.",
    modern: "\"Zionist Jews rule Washington, media, and entertainment.\"",
  },
  {
    num: 6, trope: "WEAPONIZED_ANTISEMITISM", title: "Dismiss Antisemitism Claims",
    soviet: "Calling criticism of Zionism antisemitic is a demagogic method.",
    modern: "\"They weaponize antisemitism to silence Palestinians.\"",
  },
  {
    num: 7, trope: "BLOOD_LIBEL", title: "Accuse of Atrocities",
    soviet: "Zionists exterminate civilians like the Hitlerites did.",
    modern: "\"Israel deliberately murders children for sport.\"",
  },
  {
    num: 8, trope: "DUAL_LOYALTY", title: "Question Jewish Loyalty",
    soviet: "Zionism makes Jews agents of a foreign power.",
    modern: "\"Their real loyalty is to Israel, not their own country.\"",
  },
  {
    num: 9, trope: "ANTI_ZIONISM_PROGRESSIVE", title: "Make It a Progressive Duty",
    soviet: "Workers, peasants, intelligentsia: expose Zionism!",
    modern: "\"As progressives, opposing Zionism is a moral obligation.\"",
  },
];

function PropagandaPlaybook({ matches }) {
  const [activeStep, setActiveStep] = useState(null);

  const tropeMatchCounts = useMemo(() => {
    const counts = {};
    for (const m of matches) {
      for (const t of (m.tropes || [])) counts[t] = (counts[t] || 0) + 1;
    }
    return counts;
  }, [matches]);

  return (
    <div>
      <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.6 }}>
        Soviet anti-Zionist propaganda followed a systematic playbook — a sequence of rhetorical moves that built upon each other.
        Modern anti-Zionist rhetoric follows the same sequence. Click each step to see the pattern.
      </p>
      <div style={{ position: "relative" }}>
        {/* Vertical connector line */}
        <div style={{ position: "absolute", left: 19, top: 16, bottom: 16, width: 2, background: "var(--border)" }} />

        {PLAYBOOK_STEPS.map((step, i) => {
          const color = TROPE_COLORS[step.trope] || "#888";
          const count = tropeMatchCounts[step.trope] || 0;
          const isActive = activeStep === i;
          return (
            <div key={i} onClick={() => setActiveStep(isActive ? null : i)}
              style={{ display: "flex", gap: 16, marginBottom: 8, cursor: "pointer", position: "relative" }}>
              {/* Step number circle */}
              <div style={{
                width: 40, height: 40, borderRadius: "50%", background: isActive ? color : "var(--bg-card-alt)",
                border: `2px solid ${color}`, display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 14, fontWeight: 700, color: isActive ? "#fff" : color, flexShrink: 0, zIndex: 1,
                transition: "all 0.2s",
              }}>
                {step.num}
              </div>

              {/* Content */}
              <div style={{
                flex: 1, background: isActive ? color + "11" : "var(--bg-card)",
                border: `1px solid ${isActive ? color + "44" : "var(--border)"}`,
                borderRadius: 8, padding: "10px 16px", transition: "all 0.2s",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: color }}>{step.title}</div>
                  {count > 0 && (
                    <span style={{ fontSize: 10, color: "var(--text-muted)", background: "var(--bg-card-alt)", padding: "1px 6px", borderRadius: 8 }}>
                      {count} match{count !== 1 ? "es" : ""}
                    </span>
                  )}
                </div>

                {isActive && (
                  <div style={{ marginTop: 10, animation: "fadeIn 0.3s ease" }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                      <div>
                        <div style={{ fontSize: 9, letterSpacing: 2, color: "#c0392b", fontWeight: 700, marginBottom: 4 }}>SOVIET TEMPLATE</div>
                        <div style={{ fontSize: 12, color: "var(--text-primary)", fontStyle: "italic", lineHeight: 1.5 }}>"{step.soviet}"</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, letterSpacing: 2, color: "#3498db", fontWeight: 700, marginBottom: 4 }}>MODERN ECHO</div>
                        <div style={{ fontSize: 12, color: "var(--text-primary)", fontStyle: "italic", lineHeight: 1.5 }}>"{step.modern}"</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// 5. SIMULATED EMBEDDING SCATTER (UMAP-like projection)
// ============================================================
function EmbeddingScatter({ matches }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [filter, setFilter] = useState("all");

  const points = useMemo(() => {
    const pts = [];
    const seed = (s) => {
      let h = 0;
      for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
      return h;
    };
    const pseudoRand = (s, offset) => {
      const v = seed(s + offset);
      return ((v % 1000) / 1000 + 1) % 1;
    };

    const tropeAngle = {};
    const allTropes = Object.keys(TROPE_COLORS);
    allTropes.forEach((t, i) => { tropeAngle[t] = (i / allTropes.length) * Math.PI * 2; });

    for (const m of matches) {
      const trope = (m.tropes || [])[0] || "ZIONISM_RACISM";
      const angle = tropeAngle[trope] || 0;
      const baseR = 0.25 + (m.ensembleScore || m.similarity) * 0.15;

      const sx = 0.5 + Math.cos(angle) * baseR + (pseudoRand(m.sovietText || "", "sx") - 0.5) * 0.15;
      const sy = 0.5 + Math.sin(angle) * baseR + (pseudoRand(m.sovietText || "", "sy") - 0.5) * 0.15;
      pts.push({
        x: sx, y: sy, era: "soviet", text: (m.sovietText || "").slice(0, 80),
        source: m.sovietSource, year: m.sovietYear, trope, matchId: m.id,
      });

      const mx = sx + (pseudoRand(m.modernText || "", "mx") - 0.5) * 0.12;
      const my = sy + (pseudoRand(m.modernText || "", "my") - 0.5) * 0.12;
      pts.push({
        x: mx, y: my, era: "modern", text: (m.modernText || "").slice(0, 80),
        source: m.modernSource, year: m.modernYear, trope, matchId: m.id,
        linkedSovietX: sx, linkedSovietY: sy,
      });
    }
    return pts;
  }, [matches]);

  const W = 600, H = 500, pad = 30;
  const scaleX = x => pad + x * (W - 2 * pad);
  const scaleY = y => pad + y * (H - 2 * pad);

  const filteredPoints = filter === "all" ? points : points.filter(p => p.trope === filter);
  const modernPts = filteredPoints.filter(p => p.era === "modern" && p.linkedSovietX != null);

  return (
    <div>
      <div style={{ display: "flex", gap: 4, marginBottom: 12, flexWrap: "wrap" }}>
        <button onClick={() => setFilter("all")} style={{ ...filterBtn, background: filter === "all" ? "var(--bg-card-alt)" : "transparent", color: filter === "all" ? "var(--text-heading)" : "var(--text-muted)" }}>All</button>
        {Object.keys(TROPE_COLORS).map(t => (
          <button key={t} onClick={() => setFilter(t)} style={{
            ...filterBtn,
            background: filter === t ? TROPE_COLORS[t] + "33" : "transparent",
            color: filter === t ? TROPE_COLORS[t] : "var(--text-muted)",
            borderColor: filter === t ? TROPE_COLORS[t] + "66" : "var(--border)",
          }}>
            {TROPE_SHORT[t]}
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxHeight: 520, background: "var(--bg-card)", borderRadius: 8, border: "1px solid var(--border)" }}>
        {/* Connection lines between matched pairs */}
        {modernPts.map((p, i) => (
          <line key={`line-${i}`}
            x1={scaleX(p.linkedSovietX)} y1={scaleY(p.linkedSovietY)}
            x2={scaleX(p.x)} y2={scaleY(p.y)}
            stroke={TROPE_COLORS[p.trope] || "#555"} strokeWidth={0.7} opacity={0.25}
            strokeDasharray="3,3"
          />
        ))}

        {/* Points */}
        {filteredPoints.map((p, i) => {
          const color = p.era === "soviet" ? "#c0392b" : "#3498db";
          const isHovered = hoveredPoint === i;
          return (
            <g key={i} onMouseEnter={() => setHoveredPoint(i)} onMouseLeave={() => setHoveredPoint(null)} style={{ cursor: "pointer" }}>
              <circle
                cx={scaleX(p.x)} cy={scaleY(p.y)} r={isHovered ? 7 : 4.5}
                fill={color} stroke={isHovered ? "#fff" : TROPE_COLORS[p.trope] || "#555"}
                strokeWidth={isHovered ? 2 : 1} opacity={isHovered ? 1 : 0.75}
                style={{ transition: "r 0.15s" }}
              />
            </g>
          );
        })}

        {/* Legend */}
        <circle cx={W - 80} cy={20} r={5} fill="#c0392b" />
        <text x={W - 70} y={24} fill="var(--text-secondary)" fontSize={10}>Soviet</text>
        <circle cx={W - 80} cy={36} r={5} fill="#3498db" />
        <text x={W - 70} y={40} fill="var(--text-secondary)" fontSize={10}>Modern</text>
      </svg>

      {hoveredPoint !== null && filteredPoints[hoveredPoint] && (
        <div style={{ padding: "8px 12px", background: "var(--bg-card-alt)", borderRadius: 6, fontSize: 12, marginTop: 8 }}>
          <span style={{ color: filteredPoints[hoveredPoint].era === "soviet" ? "#c0392b" : "#3498db", fontWeight: 700 }}>
            {filteredPoints[hoveredPoint].era === "soviet" ? "SOVIET" : "MODERN"} ({filteredPoints[hoveredPoint].year})
          </span>
          <span style={{ color: "var(--text-muted)", margin: "0 6px" }}>•</span>
          <span style={{ color: TROPE_COLORS[filteredPoints[hoveredPoint].trope], fontSize: 10 }}>
            {TROPE_SHORT[filteredPoints[hoveredPoint].trope]}
          </span>
          <div style={{ color: "var(--text-primary)", marginTop: 4, fontStyle: "italic" }}>
            "{filteredPoints[hoveredPoint].text}…"
          </div>
        </div>
      )}

      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 8, fontStyle: "italic" }}>
        Simulated 2D projection — positions approximate trope clusters. Real UMAP projection requires running embeddings through umap-learn.
        Dashed lines connect matched Soviet↔Modern pairs. Nearby points share similar rhetoric.
      </div>
    </div>
  );
}

// ============================================================
// SHARED STYLES
// ============================================================
const navBtn = {
  background: "var(--bg-card-alt)", border: "1px solid var(--border)", color: "var(--text-secondary)",
  padding: "4px 12px", borderRadius: 6, cursor: "pointer", fontSize: 11, fontFamily: "inherit",
};

const filterBtn = {
  border: "1px solid var(--border)", padding: "2px 8px", borderRadius: 4,
  cursor: "pointer", fontSize: 10, fontFamily: "inherit",
};

// ============================================================
// MAIN PROTOTYPES SHELL
// ============================================================
const VIEWS = [
  { id: "highlight", label: "Text Highlighting", icon: "✦" },
  { id: "sankey", label: "Rhetoric Flow", icon: "⇢" },
  { id: "network", label: "Echo Network", icon: "◎" },
  { id: "playbook", label: "Propaganda Playbook", icon: "▶" },
  { id: "scatter", label: "Embedding Space", icon: "⊚" },
];

export default function VizPrototypes() {
  const data = useMatchData();
  const [activeView, setActiveView] = useState(0);

  if (!data) {
    return (
      <div style={{ minHeight: "100vh", background: "var(--bg-page)", color: "var(--text-primary)", fontFamily: "'Courier New', 'Consolas', monospace", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 20, marginBottom: 8 }}>Loading visualization data...</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Fetching viz_data.json</div>
        </div>
      </div>
    );
  }

  const matches = data.matches || [];

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-page)", color: "var(--text-primary)", fontFamily: "'Courier New', 'Consolas', monospace", padding: "24px 16px" }}>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
      <div style={{ maxWidth: 940, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ fontSize: 10, letterSpacing: 4, color: "#c0392b", textTransform: "uppercase", marginBottom: 4 }}>
            VISUALIZATION PROTOTYPES
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 300, color: "var(--text-heading)", margin: 0, fontFamily: "Georgia, serif" }}>
            I've Seen This Before
          </h1>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "8px auto 0", maxWidth: 500 }}>
            5 experimental visualizations using {matches.length} matches across {data.soviet_corpus_size || "~1,900"} Soviet and {data.modern_corpus_size || "~2,000"} modern passages
          </p>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
            <a href="/" style={{ color: "var(--text-muted)" }}>← Back to main app</a>
          </div>
        </div>

        {/* View Selector */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid var(--border)", paddingBottom: 1, overflowX: "auto" }}>
          {VIEWS.map((v, i) => (
            <button key={v.id} onClick={() => setActiveView(i)} style={{
              background: activeView === i ? "var(--bg-card-alt)" : "transparent",
              border: "none", color: activeView === i ? "var(--text-heading)" : "var(--text-muted)",
              padding: "10px 14px", cursor: "pointer", fontSize: 12, fontFamily: "inherit",
              fontWeight: activeView === i ? 700 : 400, borderRadius: "8px 8px 0 0", whiteSpace: "nowrap",
            }}>
              {v.icon} {v.label}
            </button>
          ))}
        </div>

        {/* View Content */}
        <div style={{
          background: "var(--gradient-card)", border: "1px solid var(--border)",
          borderRadius: 12, padding: 24, boxShadow: "var(--shadow)",
        }}>
          {/* Description bar */}
          <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 16, lineHeight: 1.6, paddingBottom: 12, borderBottom: "1px solid var(--border)" }}>
            {activeView === 0 && "Side-by-side text comparison with shared vocabulary and trope keywords highlighted. Shows the specific words and phrases that make Soviet and modern texts echo each other."}
            {activeView === 1 && "Sankey flow diagram showing how propaganda flows from Soviet source documents through trope categories to modern source types. Band width = number of matches. Hover links for details."}
            {activeView === 2 && "Bipartite network graph connecting Soviet passages (left, red) to modern echoes (right, blue). Edge color = primary trope. Hover nodes to isolate connections."}
            {activeView === 3 && "The 9-step Soviet propaganda playbook — not random talking points, but a systematic rhetorical strategy. Modern rhetoric follows the same sequence. Click steps to expand."}
            {activeView === 4 && "Simulated embedding space projection. Each dot is a text passage; nearby dots have similar rhetoric. Soviet (red) and Modern (blue) passages that echo each other cluster together. Filter by trope."}
          </div>

          {activeView === 0 && <HighlightedDiff matches={matches} />}
          {activeView === 1 && <SankeyFlow matches={matches} />}
          {activeView === 2 && <NetworkGraph matches={matches} />}
          {activeView === 3 && <PropagandaPlaybook matches={matches} />}
          {activeView === 4 && <EmbeddingScatter matches={matches} />}
        </div>

        <div style={{ textAlign: "center", marginTop: 24, fontSize: 10, color: "var(--text-muted)", letterSpacing: 2 }}>
          PROTOTYPE VISUALIZATIONS • BUILT WITH REACT + SVG • DATA FROM ML PIPELINE
        </div>
      </div>
    </div>
  );
}
