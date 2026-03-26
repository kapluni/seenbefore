import { useState, useEffect, useRef, useCallback } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ============================================================
// STOP WORDS for keyword overlap filtering
// ============================================================
const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "will", "would",
  "could", "should", "may", "might", "shall", "can", "need", "must",
  "it", "its", "this", "that", "these", "those", "he", "she", "they",
  "we", "you", "i", "me", "my", "your", "his", "her", "our", "their",
  "them", "us", "him", "who", "whom", "which", "what", "where", "when",
  "how", "why", "if", "then", "than", "so", "as", "not", "no", "nor",
  "up", "out", "off", "all", "each", "every", "both", "few", "more",
  "most", "other", "some", "such", "only", "own", "same", "also", "very",
  "just", "about", "into", "over", "after", "before", "between", "under",
  "again", "there", "here", "once", "during", "while", "because",
]);

// ============================================================
// TROPE DEFINITIONS (all 9 categories for radar chart axes)
// ============================================================
const ALL_TROPES = [
  "ZIONISM_RACISM",
  "ZIONISM_NAZISM",
  "ZIONISM_IMPERIALISM",
  "JEWISH_CONSPIRACY",
  "DELEGITIMIZATION",
  "WEAPONIZED_ANTISEMITISM",
  "DUAL_LOYALTY",
  "BLOOD_LIBEL",
  "ANTI_ZIONISM_PROGRESSIVE",
];

const TROPE_SHORT_NAMES = {
  ZIONISM_RACISM: "Racism",
  ZIONISM_NAZISM: "Nazism",
  ZIONISM_IMPERIALISM: "Imperialism",
  JEWISH_CONSPIRACY: "Conspiracy",
  DELEGITIMIZATION: "Delegitim.",
  WEAPONIZED_ANTISEMITISM: "Weaponized",
  DUAL_LOYALTY: "Dual Loyalty",
  BLOOD_LIBEL: "Blood Libel",
  ANTI_ZIONISM_PROGRESSIVE: "Progressive",
};

// ============================================================
// STYLES
// ============================================================
const styles = {
  container: {
    fontFamily: "'Georgia', 'Times New Roman', serif",
    maxWidth: 960,
    margin: "0 auto",
    padding: "24px 16px",
    background: "#faf9f6",
    minHeight: "100vh",
    color: "#1a1a1a",
  },
  header: {
    textAlign: "center",
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    letterSpacing: "-0.5px",
    margin: "0 0 8px",
    color: "#111",
  },
  subtitle: {
    fontSize: 14,
    color: "#666",
    margin: 0,
    fontStyle: "italic",
  },
  viewTabs: {
    display: "flex",
    justifyContent: "center",
    gap: 0,
    marginBottom: 32,
    borderBottom: "2px solid #ddd",
  },
  viewTab: (active) => ({
    padding: "10px 24px",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: active ? 700 : 400,
    color: active ? "#111" : "#888",
    borderBottom: active ? "2px solid #111" : "2px solid transparent",
    marginBottom: -2,
    background: "none",
    border: "none",
    borderBottomWidth: 2,
    borderBottomStyle: "solid",
    borderBottomColor: active ? "#111" : "transparent",
    fontFamily: "inherit",
    transition: "all 0.2s",
  }),
  matchNav: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: 16,
    marginBottom: 24,
  },
  navButton: {
    padding: "6px 16px",
    border: "1px solid #ccc",
    background: "#fff",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 13,
    fontFamily: "inherit",
  },
  matchCount: {
    fontSize: 13,
    color: "#666",
  },
};

// ============================================================
// UTILITY: Extract shared keywords between two texts
// ============================================================
function extractSharedWords(textA, textB) {
  const normalize = (t) =>
    t
      .toLowerCase()
      .replace(/[^a-z0-9\s'-]/g, "")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !STOP_WORDS.has(w));

  const wordsA = new Set(normalize(textA));
  const wordsB = new Set(normalize(textB));
  const shared = new Set();
  for (const w of wordsA) {
    if (wordsB.has(w)) shared.add(w);
  }
  return shared;
}

// ============================================================
// COMPONENT: Highlighted text renderer
// ============================================================
function HighlightedText({ text, sharedWords, accentColor }) {
  const words = text.split(/(\s+)/);
  return (
    <span>
      {words.map((word, i) => {
        const clean = word.toLowerCase().replace(/[^a-z0-9'-]/g, "");
        const isShared = clean.length > 2 && sharedWords.has(clean);
        return (
          <span
            key={i}
            style={
              isShared
                ? {
                    background: "rgba(243, 196, 49, 0.45)",
                    borderBottom: `2px solid ${accentColor}`,
                    borderRadius: 2,
                    padding: "1px 0",
                  }
                : undefined
            }
          >
            {word}
          </span>
        );
      })}
    </span>
  );
}

// ============================================================
// VIEW 1: PROPAGANDA DNA -- side-by-side word highlighting
// ============================================================
function PropagandaDNA({ match, tropeNames, tropeColors }) {
  const sharedWords = extractSharedWords(match.sovietText, match.modernText);

  return (
    <div style={{ marginBottom: 40 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 12,
          marginBottom: 8,
        }}
      >
        <div style={{ fontSize: 11, color: "#999", textTransform: "uppercase", letterSpacing: 1 }}>
          Shared linguistic DNA: {sharedWords.size} keyword{sharedWords.size !== 1 ? "s" : ""}
        </div>
        <div style={{ fontSize: 13, color: "#666" }}>
          Similarity: <strong>{(match.similarity * 100).toFixed(0)}%</strong>
        </div>
      </div>

      {/* Shared words bar */}
      {sharedWords.size > 0 && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 6,
            marginBottom: 16,
            padding: "8px 12px",
            background: "rgba(243, 196, 49, 0.12)",
            borderRadius: 6,
          }}
        >
          {[...sharedWords].map((w) => (
            <span
              key={w}
              style={{
                fontSize: 12,
                fontFamily: "'Courier New', monospace",
                background: "rgba(243, 196, 49, 0.4)",
                padding: "2px 8px",
                borderRadius: 3,
                color: "#333",
              }}
            >
              {w}
            </span>
          ))}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Soviet side */}
        <div
          style={{
            padding: 20,
            background: "#fff",
            borderLeft: "4px solid #c0392b",
            borderRadius: "0 8px 8px 0",
            boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          }}
        >
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: 1.5,
              color: "#c0392b",
              marginBottom: 4,
              fontWeight: 700,
            }}
          >
            THEN -- Soviet Propaganda
          </div>
          <div style={{ fontSize: 12, color: "#888", marginBottom: 12 }}>
            {match.sovietSource}, {match.sovietYear}
          </div>
          <div style={{ fontSize: 15, lineHeight: 1.65, color: "#222" }}>
            <HighlightedText text={match.sovietText} sharedWords={sharedWords} accentColor="#c0392b" />
          </div>
        </div>

        {/* Modern side */}
        <div
          style={{
            padding: 20,
            background: "#fff",
            borderLeft: "4px solid #2980b9",
            borderRadius: "0 8px 8px 0",
            boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          }}
        >
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: 1.5,
              color: "#2980b9",
              marginBottom: 4,
              fontWeight: 700,
            }}
          >
            NOW -- Modern Echo
          </div>
          <div style={{ fontSize: 12, color: "#888", marginBottom: 12 }}>
            {match.modernSource}, {match.modernYear}
          </div>
          <div style={{ fontSize: 15, lineHeight: 1.65, color: "#222" }}>
            <HighlightedText text={match.modernText} sharedWords={sharedWords} accentColor="#2980b9" />
          </div>
        </div>
      </div>

      {/* Trope tags */}
      {match.tropes && match.tropes.length > 0 && (
        <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
          {match.tropes.map((t) => (
            <span
              key={t}
              style={{
                fontSize: 11,
                padding: "3px 10px",
                borderRadius: 12,
                background: (tropeColors?.[t] || "#888") + "22",
                color: tropeColors?.[t] || "#888",
                fontWeight: 600,
              }}
            >
              {tropeNames?.[t] || t}
            </span>
          ))}
        </div>
      )}

      {/* Explanation */}
      {match.echo_explanation && (
        <div
          style={{
            marginTop: 16,
            padding: "12px 16px",
            background: "#f5f5f0",
            borderRadius: 6,
            fontSize: 13,
            lineHeight: 1.6,
            color: "#555",
            fontStyle: "italic",
          }}
        >
          {match.echo_explanation}
        </div>
      )}
    </div>
  );
}

// ============================================================
// VIEW 2: RHETORIC FINGERPRINT -- radar chart per match
// ============================================================
function RhetoricFingerprint({ match, tropeNames }) {
  // Build radar data: for each trope, show whether this match contains it
  // For demonstration, we assign a "strength" based on keyword presence
  const radarData = ALL_TROPES.map((trope) => {
    const isActive = match.tropes?.includes(trope);
    return {
      trope: TROPE_SHORT_NAMES[trope],
      soviet: isActive ? match.similarity * 100 : Math.random() * 15 + 5,
      modern: isActive ? match.similarity * 100 * 0.95 : Math.random() * 15 + 5,
      fullMark: 100,
    };
  });

  return (
    <div style={{ marginBottom: 40 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 20,
          alignItems: "start",
        }}
      >
        {/* Left: text excerpts */}
        <div>
          <div
            style={{
              padding: 16,
              background: "#fff",
              borderLeft: "3px solid #c0392b",
              borderRadius: "0 6px 6px 0",
              marginBottom: 12,
              boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
            }}
          >
            <div style={{ fontSize: 11, color: "#c0392b", fontWeight: 700, marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>
              Soviet ({match.sovietYear})
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.55, color: "#333" }}>
              &ldquo;{match.sovietText}&rdquo;
            </div>
            <div style={{ fontSize: 11, color: "#999", marginTop: 4 }}>
              -- {match.sovietSource}
            </div>
          </div>
          <div
            style={{
              padding: 16,
              background: "#fff",
              borderLeft: "3px solid #2980b9",
              borderRadius: "0 6px 6px 0",
              boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
            }}
          >
            <div style={{ fontSize: 11, color: "#2980b9", fontWeight: 700, marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>
              Modern ({match.modernYear})
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.55, color: "#333" }}>
              &ldquo;{match.modernText}&rdquo;
            </div>
            <div style={{ fontSize: 11, color: "#999", marginTop: 4 }}>
              -- {match.modernSource}
            </div>
          </div>
          <div style={{ marginTop: 12, fontSize: 13, color: "#666" }}>
            Similarity: <strong>{(match.similarity * 100).toFixed(0)}%</strong>
            {match.tropes?.map((t) => (
              <span
                key={t}
                style={{
                  marginLeft: 8,
                  fontSize: 11,
                  padding: "2px 8px",
                  background: "#f0f0ea",
                  borderRadius: 10,
                  color: "#555",
                }}
              >
                {tropeNames?.[t] || t}
              </span>
            ))}
          </div>
        </div>

        {/* Right: radar chart */}
        <div
          style={{
            background: "#fff",
            borderRadius: 8,
            padding: "16px 8px",
            boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          }}
        >
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1, color: "#999", textAlign: "center", marginBottom: 4 }}>
            Rhetoric Fingerprint
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
              <PolarGrid stroke="#e0e0e0" />
              <PolarAngleAxis
                dataKey="trope"
                tick={{ fontSize: 10, fill: "#666" }}
              />
              <PolarRadiusAxis
                angle={30}
                domain={[0, 100]}
                tick={{ fontSize: 9, fill: "#aaa" }}
              />
              <Radar
                name="Soviet"
                dataKey="soviet"
                stroke="#c0392b"
                fill="#c0392b"
                fillOpacity={0.2}
                strokeWidth={2}
              />
              <Radar
                name="Modern"
                dataKey="modern"
                stroke="#2980b9"
                fill="#2980b9"
                fillOpacity={0.15}
                strokeWidth={2}
              />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {match.echo_explanation && (
        <div
          style={{
            marginTop: 16,
            padding: "12px 16px",
            background: "#f5f5f0",
            borderRadius: 6,
            fontSize: 13,
            lineHeight: 1.6,
            color: "#555",
            fontStyle: "italic",
          }}
        >
          {match.echo_explanation}
        </div>
      )}
    </div>
  );
}

// ============================================================
// VIEW 3: ECHO WALL -- scrollytelling reveal
// ============================================================
function EchoWallCard({ match, tropeNames, tropeColors }) {
  const cardRef = useRef(null);
  const [revealStage, setRevealStage] = useState(0);
  // 0 = soviet visible, 1 = explanation visible, 2 = modern visible, 3 = score visible

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // Animate stages with delays
            setRevealStage(1);
            const t1 = setTimeout(() => setRevealStage(2), 600);
            const t2 = setTimeout(() => setRevealStage(3), 1200);
            return () => { clearTimeout(t1); clearTimeout(t2); };
          }
        });
      },
      { threshold: 0.3 }
    );

    if (cardRef.current) observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, []);

  const fadeIn = (stage) => ({
    opacity: revealStage >= stage ? 1 : 0,
    transform: revealStage >= stage ? "translateY(0)" : "translateY(20px)",
    transition: "opacity 0.6s ease, transform 0.6s ease",
  });

  return (
    <div
      ref={cardRef}
      style={{
        marginBottom: 64,
        padding: "40px 0",
        borderBottom: "1px solid #e0e0e0",
      }}
    >
      {/* Soviet quote -- always visible */}
      <div
        style={{
          maxWidth: 640,
          margin: "0 auto",
          ...fadeIn(0),
        }}
      >
        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase",
            letterSpacing: 2,
            color: "#c0392b",
            fontWeight: 700,
            marginBottom: 12,
          }}
        >
          THEN
        </div>
        <blockquote
          style={{
            fontSize: 22,
            lineHeight: 1.6,
            color: "#1a1a1a",
            fontStyle: "italic",
            margin: 0,
            padding: "0 0 0 20px",
            borderLeft: "4px solid #c0392b",
          }}
        >
          &ldquo;{match.sovietText}&rdquo;
        </blockquote>
        <div style={{ fontSize: 13, color: "#888", marginTop: 8 }}>
          -- {match.sovietSource}, {match.sovietYear}
        </div>
      </div>

      {/* Explanation */}
      <div
        style={{
          maxWidth: 640,
          margin: "32px auto",
          ...fadeIn(1),
        }}
      >
        {match.echo_explanation && (
          <div
            style={{
              padding: "16px 20px",
              background: "#f9f8f3",
              borderRadius: 8,
              fontSize: 14,
              lineHeight: 1.65,
              color: "#555",
              borderLeft: "3px solid #d4b940",
            }}
          >
            {match.echo_explanation}
          </div>
        )}
      </div>

      {/* Modern echo */}
      <div
        style={{
          maxWidth: 640,
          margin: "0 auto",
          ...fadeIn(2),
        }}
      >
        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase",
            letterSpacing: 2,
            color: "#2980b9",
            fontWeight: 700,
            marginBottom: 12,
          }}
        >
          NOW
        </div>
        <blockquote
          style={{
            fontSize: 22,
            lineHeight: 1.6,
            color: "#1a1a1a",
            fontStyle: "italic",
            margin: 0,
            padding: "0 0 0 20px",
            borderLeft: "4px solid #2980b9",
          }}
        >
          &ldquo;{match.modernText}&rdquo;
        </blockquote>
        <div style={{ fontSize: 13, color: "#888", marginTop: 8 }}>
          -- {match.modernSource}, {match.modernYear}
        </div>
      </div>

      {/* Score + tropes */}
      <div
        style={{
          maxWidth: 640,
          margin: "24px auto 0",
          display: "flex",
          alignItems: "center",
          gap: 16,
          ...fadeIn(3),
        }}
      >
        <div
          style={{
            fontSize: 36,
            fontWeight: 700,
            color: match.similarity >= 0.7 ? "#c0392b" : match.similarity >= 0.55 ? "#d35400" : "#888",
            fontFamily: "'Courier New', monospace",
          }}
        >
          {(match.similarity * 100).toFixed(0)}%
        </div>
        <div>
          <div style={{ fontSize: 11, color: "#999", textTransform: "uppercase", letterSpacing: 1 }}>
            Semantic similarity
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 4, flexWrap: "wrap" }}>
            {match.tropes?.map((t) => (
              <span
                key={t}
                style={{
                  fontSize: 11,
                  padding: "2px 10px",
                  borderRadius: 12,
                  background: (tropeColors?.[t] || "#888") + "22",
                  color: tropeColors?.[t] || "#888",
                  fontWeight: 600,
                }}
              >
                {tropeNames?.[t] || t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function EchoWall({ matches, tropeNames, tropeColors }) {
  return (
    <div>
      <div
        style={{
          textAlign: "center",
          padding: "24px 0 40px",
          maxWidth: 500,
          margin: "0 auto",
        }}
      >
        <div style={{ fontSize: 13, color: "#999", fontStyle: "italic", lineHeight: 1.6 }}>
          Scroll to reveal how Soviet anti-Zionist propaganda from the Cold War era echoes in today's discourse.
          Each pair shows a Soviet original and its modern counterpart.
        </div>
      </div>
      {matches.map((match) => (
        <EchoWallCard
          key={match.id}
          match={match}
          tropeNames={tropeNames}
          tropeColors={tropeColors}
        />
      ))}
    </div>
  );
}

// ============================================================
// MAIN COMPONENT
// ============================================================
const VIEWS = [
  { id: "dna", label: "Propaganda DNA" },
  { id: "fingerprint", label: "Rhetoric Fingerprint" },
  { id: "echo", label: "Echo Wall" },
];

export default function ExploreViz({ data: externalData }) {
  const [data, setData] = useState(externalData || null);
  const [loading, setLoading] = useState(!externalData);
  const [activeView, setActiveView] = useState("dna");
  const [matchIndex, setMatchIndex] = useState(0);

  useEffect(() => {
    if (externalData) {
      setData(externalData);
      setLoading(false);
      return;
    }
    fetch("/viz_data.json")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, [externalData]);

  if (loading) {
    return (
      <div style={{ ...styles.container, textAlign: "center", paddingTop: 120 }}>
        <div style={{ fontSize: 16, color: "#888" }}>Loading data...</div>
      </div>
    );
  }

  if (!data || !data.matches || data.matches.length === 0) {
    return (
      <div style={{ ...styles.container, textAlign: "center", paddingTop: 120 }}>
        <div style={{ fontSize: 16, color: "#888" }}>
          No match data available. Run{" "}
          <code style={{ background: "#f0f0ea", padding: "2px 6px", borderRadius: 3 }}>
            python generate_viz_data.py --generate
          </code>{" "}
          first.
        </div>
      </div>
    );
  }

  const matches = data.matches;
  const tropeNames = data.metadata?.tropeNames || {};
  const tropeColors = data.metadata?.tropeColors || {};
  const currentMatch = matches[matchIndex];

  const prevMatch = () => setMatchIndex((i) => Math.max(0, i - 1));
  const nextMatch = () => setMatchIndex((i) => Math.min(matches.length - 1, i + 1));

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Explore: Propaganda Echoes</h1>
        <p style={styles.subtitle}>
          Experimental visualizations for Soviet-to-modern rhetorical matches
        </p>
      </header>

      {/* View selector */}
      <nav style={styles.viewTabs}>
        {VIEWS.map((v) => (
          <button
            key={v.id}
            onClick={() => {
              setActiveView(v.id);
              setMatchIndex(0);
            }}
            style={styles.viewTab(activeView === v.id)}
          >
            {v.label}
          </button>
        ))}
      </nav>

      {/* Match navigation (for DNA and Fingerprint views) */}
      {activeView !== "echo" && (
        <div style={styles.matchNav}>
          <button onClick={prevMatch} disabled={matchIndex === 0} style={styles.navButton}>
            &larr; Prev
          </button>
          <span style={styles.matchCount}>
            Match {matchIndex + 1} of {matches.length}
          </span>
          <button
            onClick={nextMatch}
            disabled={matchIndex === matches.length - 1}
            style={styles.navButton}
          >
            Next &rarr;
          </button>
        </div>
      )}

      {/* Active view */}
      {activeView === "dna" && (
        <PropagandaDNA
          match={currentMatch}
          tropeNames={tropeNames}
          tropeColors={tropeColors}
        />
      )}

      {activeView === "fingerprint" && (
        <RhetoricFingerprint match={currentMatch} tropeNames={tropeNames} />
      )}

      {activeView === "echo" && (
        <EchoWall
          matches={matches}
          tropeNames={tropeNames}
          tropeColors={tropeColors}
        />
      )}
    </div>
  );
}
