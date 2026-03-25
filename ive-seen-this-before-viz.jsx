import { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from "recharts";

const MATCH_DATA = [
  {
    id: 1,
    sovietText: "The main posits of modern Zionism are militant chauvinism, racism, anti-Communism and anti-Sovietism.",
    sovietSource: "Great Soviet Encyclopedia",
    sovietYear: 1975,
    modernText: "Zionism is a racist, settler-colonial ideology that promotes Jewish supremacy over indigenous Palestinians.",
    modernSource: "Campus BDS Resolution",
    modernYear: 2024,
    similarity: 0.91,
    tropes: ["ZIONISM_RACISM", "ZIONISM_IMPERIALISM"],
    confidence: "high",
  },
  {
    id: 2,
    sovietText: "Absurd are attempts of Zionist ideologists to present those who criticize them as antisemitic.",
    sovietSource: "Anti-Zionist Committee Declaration, Pravda",
    sovietYear: 1983,
    modernText: "Accusations of antisemitism are weaponized to silence legitimate criticism of Israel's apartheid regime.",
    modernSource: "Social Media Post",
    modernYear: 2024,
    similarity: 0.88,
    tropes: ["WEAPONIZED_ANTISEMITISM"],
    confidence: "high",
  },
  {
    id: 3,
    sovietText: "International Zionist Organization influences or controls significant part of media agencies and outlets in the West.",
    sovietSource: "Great Soviet Encyclopedia",
    sovietYear: 1975,
    modernText: "The Zionist lobby controls the mainstream media narrative and uses its financial influence to suppress Palestinian voices.",
    modernSource: "Protest Speech",
    modernYear: 2023,
    similarity: 0.87,
    tropes: ["JEWISH_CONSPIRACY"],
    confidence: "high",
  },
  {
    id: 4,
    sovietText: "Serving as the front squad of colonialism and neo-colonialism, international Zionism actively participates in the fight against national liberation movements.",
    sovietSource: "Great Soviet Encyclopedia",
    sovietYear: 1975,
    modernText: "Israel is a settler-colonial state built on the dispossession of indigenous Palestinians. Zionism is a tool of Western imperialism.",
    modernSource: "Academic Paper",
    modernYear: 2024,
    similarity: 0.86,
    tropes: ["ZIONISM_IMPERIALISM"],
    confidence: "high",
  },
  {
    id: 5,
    sovietText: "The Hitlerites acted in the same way when they exterminated the inferior Jewish race. Zionism-Fascism-Hitlerism.",
    sovietSource: "Leningradskaya Pravda / Anti-Zionist Committee",
    sovietYear: 1983,
    modernText: "What Israel is doing in Gaza is genocide — they are the new Nazis carrying out a holocaust against Palestinians.",
    modernSource: "Campus Protest Chant",
    modernYear: 2024,
    similarity: 0.84,
    tropes: ["ZIONISM_NAZISM", "BLOOD_LIBEL"],
    confidence: "medium",
  },
  {
    id: 6,
    sovietText: "We call on all Soviet citizens: workers, peasants, representatives of intelligentsia: take active part in exposing Zionism.",
    sovietSource: "Anti-Zionist Committee Declaration",
    sovietYear: 1983,
    modernText: "As progressives, we have a moral obligation to stand against Zionism and support the Palestinian liberation struggle through BDS.",
    modernSource: "Student Organization Statement",
    modernYear: 2024,
    similarity: 0.79,
    tropes: ["ANTI_ZIONISM_PROGRESSIVE"],
    confidence: "medium",
  },
];

const NEGATIVE_EXAMPLES = [
  {
    text: "I disagree with the Israeli government's settlement expansion policy.",
    similarity: 0.31,
    label: "Policy Criticism",
  },
  {
    text: "The two-state solution requires compromises from both sides.",
    similarity: 0.22,
    label: "Peace Advocacy",
  },
];

const TROPE_NAMES = {
  ZIONISM_RACISM: "Zionism = Racism",
  ZIONISM_NAZISM: "Zionism = Nazism",
  ZIONISM_IMPERIALISM: "Zionism = Imperialism",
  JEWISH_CONSPIRACY: "Zionist Conspiracy",
  DELEGITIMIZATION: "Delegitimization",
  WEAPONIZED_ANTISEMITISM: "Weaponized Antisemitism",
  DUAL_LOYALTY: "Dual Loyalty",
  BLOOD_LIBEL: "Blood Libel / Atrocity",
  ANTI_ZIONISM_PROGRESSIVE: "Anti-Zionism as Duty",
};

const TROPE_COLORS = {
  ZIONISM_RACISM: "#c0392b",
  ZIONISM_NAZISM: "#8e44ad",
  ZIONISM_IMPERIALISM: "#2980b9",
  JEWISH_CONSPIRACY: "#d35400",
  DELEGITIMIZATION: "#27ae60",
  WEAPONIZED_ANTISEMITISM: "#f39c12",
  DUAL_LOYALTY: "#1abc9c",
  BLOOD_LIBEL: "#e74c3c",
  ANTI_ZIONISM_PROGRESSIVE: "#3498db",
};

const TIMELINE_EVENTS = [
  { year: 1963, label: "Kichko: 'Judaism Without Embellishment'", type: "soviet" },
  { year: 1967, label: "Six-Day War → massive anti-Zionist campaign", type: "soviet" },
  { year: 1969, label: "Ivanov: 'Caution: Zionism!' — 800K copies", type: "soviet" },
  { year: 1975, label: "UN Resolution 3379: 'Zionism is Racism'", type: "soviet" },
  { year: 1983, label: "Anti-Zionist Committee of Soviet Public", type: "soviet" },
  { year: 1991, label: "UN repeals Resolution 3379", type: "transition" },
  { year: 2001, label: "Durban Conference revives 'Zionism = Racism'", type: "modern" },
  { year: 2005, label: "BDS Movement founded", type: "modern" },
  { year: 2016, label: "Campus BDS resolutions accelerate", type: "modern" },
  { year: 2023, label: "Post-Oct 7: 'settler-colonial' rhetoric surges", type: "modern" },
  { year: 2024, label: "Campus encampments; Soviet tropes go mainstream", type: "modern" },
];

// ----- COMPONENTS -----

function ConfidenceBadge({ level }) {
  const styles = {
    high: { bg: "#1a472a", border: "#2ecc71", text: "#2ecc71" },
    medium: { bg: "#4a3800", border: "#f1c40f", text: "#f1c40f" },
    low: { bg: "#4a1a1a", border: "#e74c3c", text: "#e74c3c" },
  };
  const s = styles[level] || styles.low;
  return (
    <span style={{
      background: s.bg, border: `1px solid ${s.border}`, color: s.text,
      padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 700,
      textTransform: "uppercase", letterSpacing: 1,
    }}>
      {level}
    </span>
  );
}

function SimilarityBar({ score }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.85 ? "#2ecc71" : score >= 0.70 ? "#f1c40f" : score >= 0.55 ? "#e67e22" : "#555";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: "#1a1a2e", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%", background: color,
          borderRadius: 3, transition: "width 1s ease-out",
        }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color, minWidth: 40, textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

function MatchCard({ match, index, isVisible }) {
  return (
    <div style={{
      background: "linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)",
      border: "1px solid #2a2a4a",
      borderRadius: 12,
      padding: 24,
      opacity: isVisible ? 1 : 0,
      transform: isVisible ? "translateY(0)" : "translateY(20px)",
      transition: `all 0.6s ease-out ${index * 0.15}s`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <ConfidenceBadge level={match.confidence} />
          {match.tropes.map(t => (
            <span key={t} style={{
              background: TROPE_COLORS[t] + "22",
              border: `1px solid ${TROPE_COLORS[t]}55`,
              color: TROPE_COLORS[t],
              padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600,
            }}>
              {TROPE_NAMES[t]}
            </span>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 60px 1fr", gap: 16, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 2, color: "#c0392b", fontWeight: 700, marginBottom: 8 }}>
            THEN — {match.sovietYear}
          </div>
          <div style={{
            fontFamily: "'Georgia', serif", fontSize: 15, lineHeight: 1.6,
            color: "#ccc", fontStyle: "italic", borderLeft: "3px solid #c0392b",
            paddingLeft: 16,
          }}>
            "{match.sovietText}"
          </div>
          <div style={{ fontSize: 11, color: "#666", marginTop: 8 }}>
            — {match.sovietSource}
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{
            width: 44, height: 44, borderRadius: "50%",
            background: `conic-gradient(${match.similarity >= 0.85 ? '#2ecc71' : '#f1c40f'} ${match.similarity * 360}deg, #1a1a2e 0deg)`,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: "50%", background: "#0d0d1a",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 11, fontWeight: 700, color: "#fff",
            }}>
              {Math.round(match.similarity * 100)}%
            </div>
          </div>
        </div>

        <div>
          <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 2, color: "#3498db", fontWeight: 700, marginBottom: 8 }}>
            NOW — {match.modernYear}
          </div>
          <div style={{
            fontFamily: "'Georgia', serif", fontSize: 15, lineHeight: 1.6,
            color: "#ccc", fontStyle: "italic", borderLeft: "3px solid #3498db",
            paddingLeft: 16,
          }}>
            "{match.modernText}"
          </div>
          <div style={{ fontSize: 11, color: "#666", marginTop: 8 }}>
            — {match.modernSource}
          </div>
        </div>
      </div>

      <SimilarityBar score={match.similarity} />
    </div>
  );
}

function TimelineView() {
  return (
    <div style={{ position: "relative", padding: "20px 0" }}>
      <div style={{
        position: "absolute", left: "50%", top: 0, bottom: 0, width: 2,
        background: "linear-gradient(to bottom, #c0392b, #666, #3498db)",
      }} />
      {TIMELINE_EVENTS.map((evt, i) => {
        const isLeft = evt.type === "soviet" || evt.type === "transition";
        const color = evt.type === "soviet" ? "#c0392b" : evt.type === "transition" ? "#f1c40f" : "#3498db";
        return (
          <div key={i} style={{
            display: "flex", alignItems: "center", marginBottom: 12,
            flexDirection: isLeft ? "row" : "row-reverse",
          }}>
            <div style={{
              flex: 1, textAlign: isLeft ? "right" : "left",
              paddingRight: isLeft ? 24 : 0, paddingLeft: isLeft ? 0 : 24,
            }}>
              <span style={{ fontSize: 12, fontWeight: 700, color }}>{evt.year}</span>
              <span style={{ fontSize: 12, color: "#999", marginLeft: 8 }}>{evt.label}</span>
            </div>
            <div style={{
              width: 12, height: 12, borderRadius: "50%", background: color,
              border: "2px solid #0d0d1a", zIndex: 1, flexShrink: 0,
            }} />
            <div style={{ flex: 1 }} />
          </div>
        );
      })}
      <div style={{ display: "flex", justifyContent: "center", gap: 24, marginTop: 16 }}>
        {[
          { color: "#c0392b", label: "Soviet Propaganda" },
          { color: "#f1c40f", label: "Transition" },
          { color: "#3498db", label: "Modern Echo" },
        ].map(l => (
          <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: l.color }} />
            <span style={{ fontSize: 11, color: "#888" }}>{l.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TropeHeatmap() {
  const tropeIds = Object.keys(TROPE_NAMES);
  const sovietCounts = {};
  const modernCounts = {};
  MATCH_DATA.forEach(m => {
    m.tropes.forEach(t => {
      sovietCounts[t] = (sovietCounts[t] || 0) + 1;
      modernCounts[t] = (modernCounts[t] || 0) + 1;
    });
  });

  const chartData = tropeIds.map(t => ({
    name: TROPE_NAMES[t],
    soviet: sovietCounts[t] || 0,
    modern: modernCounts[t] || 0,
  })).filter(d => d.soviet > 0 || d.modern > 0).sort((a, b) => (b.soviet + b.modern) - (a.soviet + a.modern));

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
        <XAxis type="number" tick={{ fill: "#888", fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={150} tick={{ fill: "#ccc", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, color: "#ccc", fontSize: 12 }}
        />
        <Bar dataKey="soviet" name="Soviet Sources" fill="#c0392b" radius={[0, 4, 4, 0]} />
        <Bar dataKey="modern" name="Modern Echoes" fill="#3498db" radius={[0, 4, 4, 0]} />
        <Legend wrapperStyle={{ fontSize: 11, color: "#888" }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ThresholdDemo() {
  const allScores = [
    ...MATCH_DATA.map(m => ({ label: m.modernSource, score: m.similarity, type: "echo" })),
    ...NEGATIVE_EXAMPLES.map(n => ({ label: n.label, score: n.similarity, type: "legitimate" })),
  ].sort((a, b) => b.score - a.score);

  return (
    <div>
      <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
        {[
          { label: "HIGH ≥0.85", color: "#2ecc71" },
          { label: "MEDIUM ≥0.70", color: "#f1c40f" },
          { label: "LOW ≥0.55", color: "#e67e22" },
          { label: "NO MATCH <0.55", color: "#555" },
        ].map(t => (
          <span key={t.label} style={{ fontSize: 10, color: t.color, fontWeight: 600, padding: "2px 6px", border: `1px solid ${t.color}33`, borderRadius: 4 }}>{t.label}</span>
        ))}
      </div>
      {allScores.map((item, i) => {
        const color = item.score >= 0.85 ? "#2ecc71" : item.score >= 0.70 ? "#f1c40f" : item.score >= 0.55 ? "#e67e22" : "#555";
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
            <span style={{
              fontSize: 11, color: item.type === "legitimate" ? "#2ecc71" : "#ccc",
              minWidth: 130, textAlign: "right",
              fontWeight: item.type === "legitimate" ? 700 : 400,
            }}>
              {item.type === "legitimate" ? "✓ " : ""}{item.label}
            </span>
            <div style={{ flex: 1, height: 8, background: "#1a1a2e", borderRadius: 4, overflow: "hidden" }}>
              <div style={{
                width: `${item.score * 100}%`, height: "100%",
                background: item.type === "legitimate" ? "#27ae6055" : color,
                borderRadius: 4,
                transition: "width 1s ease-out",
              }} />
            </div>
            <span style={{ fontSize: 12, fontWeight: 700, color, minWidth: 36, textAlign: "right" }}>
              {Math.round(item.score * 100)}%
            </span>
          </div>
        );
      })}
      <div style={{ fontSize: 11, color: "#27ae60", marginTop: 12, fontStyle: "italic" }}>
        ✓ Legitimate criticism correctly scores well below threshold — no false positives
      </div>
    </div>
  );
}

function SocialCard({ match }) {
  return (
    <div style={{
      background: "#0a0a14",
      border: "1px solid #2a2a4a",
      borderRadius: 16,
      padding: 24,
      maxWidth: 480,
      margin: "0 auto",
    }}>
      <div style={{ fontSize: 10, letterSpacing: 3, color: "#888", textTransform: "uppercase", textAlign: "center", marginBottom: 20 }}>
        #IveSeenThisBefore
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#c0392b", letterSpacing: 2, marginBottom: 8 }}>
            THEN — {match.sovietYear}
          </div>
          <div style={{ fontSize: 14, color: "#ccc", fontStyle: "italic", lineHeight: 1.5, borderLeft: "2px solid #c0392b", paddingLeft: 12 }}>
            "{match.sovietText}"
          </div>
          <div style={{ fontSize: 10, color: "#666", marginTop: 8 }}>
            {match.sovietSource}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#3498db", letterSpacing: 2, marginBottom: 8 }}>
            NOW — {match.modernYear}
          </div>
          <div style={{ fontSize: 14, color: "#ccc", fontStyle: "italic", lineHeight: 1.5, borderLeft: "2px solid #3498db", paddingLeft: 12 }}>
            "{match.modernText}"
          </div>
          <div style={{ fontSize: 10, color: "#666", marginTop: 8 }}>
            {match.modernSource}
          </div>
        </div>
      </div>
      <div style={{ textAlign: "center", marginTop: 20 }}>
        <div style={{
          display: "inline-block",
          background: `conic-gradient(#f1c40f ${match.similarity * 360}deg, #1a1a2e 0deg)`,
          width: 56, height: 56, borderRadius: "50%",
          padding: 3,
        }}>
          <div style={{
            width: "100%", height: "100%", borderRadius: "50%", background: "#0a0a14",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 800, color: "#f1c40f",
          }}>
            {Math.round(match.similarity * 100)}%
          </div>
        </div>
        <div style={{ fontSize: 10, color: "#666", marginTop: 4 }}>Semantic Similarity</div>
      </div>
    </div>
  );
}

// ----- MAIN APP -----

const TABS = ["Matches", "Timeline", "Tropes", "Calibration", "Social Cards"];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [visible, setVisible] = useState(false);
  const [selectedCard, setSelectedCard] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(t);
  }, [activeTab]);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#08080f",
      color: "#e0e0e0",
      fontFamily: "'Courier New', 'Consolas', monospace",
      padding: "32px 24px",
    }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ fontSize: 11, letterSpacing: 6, color: "#c0392b", textTransform: "uppercase", marginBottom: 8 }}>
            Project Prototype
          </div>
          <h1 style={{
            fontSize: 32, fontWeight: 300, color: "#fff", margin: 0,
            fontFamily: "'Georgia', serif", letterSpacing: 1,
          }}>
            I've Seen This Before
          </h1>
          <p style={{ fontSize: 13, color: "#888", maxWidth: 600, margin: "12px auto 0", lineHeight: 1.6 }}>
            Mapping Soviet anti-Zionist propaganda to its modern echoes —
            using semantic similarity to reveal the origins of today's rhetoric
          </p>
        </div>

        {/* Tabs */}
        <div style={{
          display: "flex", gap: 2, marginBottom: 32, borderBottom: "1px solid #1a1a2e",
          paddingBottom: 1, overflowX: "auto",
        }}>
          {TABS.map((tab, i) => (
            <button
              key={tab}
              onClick={() => { setVisible(false); setTimeout(() => { setActiveTab(i); setVisible(true); }, 200); }}
              style={{
                background: activeTab === i ? "#1a1a2e" : "transparent",
                border: "none",
                color: activeTab === i ? "#fff" : "#666",
                padding: "10px 20px",
                cursor: "pointer",
                fontSize: 12,
                fontFamily: "inherit",
                fontWeight: activeTab === i ? 700 : 400,
                borderRadius: "8px 8px 0 0",
                transition: "all 0.3s",
                whiteSpace: "nowrap",
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ opacity: visible ? 1 : 0, transition: "opacity 0.4s" }}>
          {activeTab === 0 && (
            <div>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 20 }}>
                Side-by-side matches between Soviet propaganda texts and modern rhetoric,
                ranked by semantic similarity score
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {MATCH_DATA.map((match, i) => (
                  <MatchCard key={match.id} match={match} index={i} isVisible={visible} />
                ))}
              </div>
            </div>
          )}

          {activeTab === 1 && (
            <div>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 20 }}>
                The pipeline from Soviet propaganda factories to modern campus discourse
              </div>
              <div style={{
                background: "linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)",
                border: "1px solid #2a2a4a", borderRadius: 12, padding: 24,
              }}>
                <TimelineView />
              </div>
            </div>
          )}

          {activeTab === 2 && (
            <div>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 20 }}>
                Which Soviet propaganda tropes appear most frequently in modern discourse
              </div>
              <div style={{
                background: "linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)",
                border: "1px solid #2a2a4a", borderRadius: 12, padding: 24,
              }}>
                <TropeHeatmap />
              </div>
            </div>
          )}

          {activeTab === 3 && (
            <div>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 20 }}>
                Confidence calibration — legitimate criticism scores well below the match threshold,
                ensuring no false positives
              </div>
              <div style={{
                background: "linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)",
                border: "1px solid #2a2a4a", borderRadius: 12, padding: 24,
              }}>
                <ThresholdDemo />
              </div>
            </div>
          )}

          {activeTab === 4 && (
            <div>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 20 }}>
                Shareable social media cards for the #IveSeenThisBefore campaign
              </div>
              <div style={{ display: "flex", gap: 8, marginBottom: 20, justifyContent: "center", flexWrap: "wrap" }}>
                {MATCH_DATA.map((m, i) => (
                  <button key={i} onClick={() => setSelectedCard(i)} style={{
                    background: selectedCard === i ? "#2a2a4a" : "transparent",
                    border: `1px solid ${selectedCard === i ? "#3498db" : "#2a2a4a"}`,
                    color: selectedCard === i ? "#fff" : "#888",
                    padding: "6px 12px", borderRadius: 6, cursor: "pointer",
                    fontSize: 11, fontFamily: "inherit",
                  }}>
                    Match {i + 1}
                  </button>
                ))}
              </div>
              <SocialCard match={MATCH_DATA[selectedCard]} />
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ textAlign: "center", marginTop: 48, paddingTop: 24, borderTop: "1px solid #1a1a2e" }}>
          <div style={{ fontSize: 10, color: "#444", letterSpacing: 2 }}>
            PROTOTYPE — DATA IS ILLUSTRATIVE • SIMILARITY SCORES WILL BE COMPUTED BY ML PIPELINE
          </div>
        </div>
      </div>
    </div>
  );
}
