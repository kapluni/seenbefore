import { useState, useEffect } from "react";
import { Scrollama, Step } from "react-scrollama";

const TROPE_LABELS = {
  ZIONISM_RACISM: "Zionism = Racism",
  ZIONISM_NAZISM: "Zionism = Nazism",
  ZIONISM_IMPERIALISM: "Zionism = Imperialism",
  JEWISH_CONSPIRACY: "Jewish Conspiracy",
  DELEGITIMIZATION: "Delegitimization",
  WEAPONIZED_ANTISEMITISM: "Weaponized Antisemitism",
  DUAL_LOYALTY: "Dual Loyalty",
  BLOOD_LIBEL: "Blood Libel",
  ANTI_ZIONISM_PROGRESSIVE: "Anti-Zionism as Duty",
};

// Identify shared significant words between two texts
function findSharedWords(text1, text2) {
  const stopWords = new Set(["the","a","an","is","are","was","were","of","and","in","to","for","that","this","it","its","with","as","by","on","at","from","or","not","but","be","has","have","had","which","their","they","them","we","our","all","also","been","more","than","other","into","can","will","would","about","out","up","no","so","if","when","do","does","did"]);
  const normalize = w => w.toLowerCase().replace(/[^a-z]/g, "");
  const words1 = new Set(text1.split(/\s+/).map(normalize).filter(w => w.length > 3 && !stopWords.has(w)));
  const words2 = new Set(text2.split(/\s+/).map(normalize).filter(w => w.length > 3 && !stopWords.has(w)));
  return new Set([...words1].filter(w => words2.has(w)));
}

function HighlightedText({ text, sharedWords, color }) {
  const words = text.split(/(\s+)/);
  return (
    <span>
      {words.map((word, i) => {
        const clean = word.toLowerCase().replace(/[^a-z]/g, "");
        const isShared = clean.length > 3 && sharedWords.has(clean);
        return (
          <span key={i} style={isShared ? {
            background: `${color}33`,
            borderBottom: `2px solid ${color}`,
            padding: "1px 0",
          } : undefined}>
            {word}
          </span>
        );
      })}
    </span>
  );
}

function ScrollMatch({ match, phase }) {
  const shared = findSharedWords(match.sovietText, match.modernText);
  const yearGap = (match.modernYear || 2024) - (match.sovietYear || 1970);

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "40px 20px",
    }}>
      <div style={{ maxWidth: 700, width: "100%" }}>
        {/* Trope badge */}
        <div style={{
          display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap",
          opacity: phase >= 0 ? 1 : 0,
          transform: phase >= 0 ? "translateY(0)" : "translateY(20px)",
          transition: "all 0.6s ease-out",
        }}>
          {match.tropes.map(t => (
            <span key={t} style={{
              fontSize: 11, fontWeight: 700, letterSpacing: 1,
              color: "#f1c40f", textTransform: "uppercase",
            }}>
              {TROPE_LABELS[t] || t}
            </span>
          ))}
        </div>

        {/* Soviet quote - always visible */}
        <div style={{
          opacity: phase >= 0 ? 1 : 0,
          transform: phase >= 0 ? "translateY(0)" : "translateY(30px)",
          transition: "all 0.8s ease-out",
        }}>
          <div style={{
            fontSize: 11, fontWeight: 700, letterSpacing: 2,
            color: "#c0392b", textTransform: "uppercase", marginBottom: 12,
          }}>
            THEN — {match.sovietYear}
          </div>
          <blockquote style={{
            fontFamily: "'Georgia', serif",
            fontSize: 22, lineHeight: 1.6, color: "#ddd",
            borderLeft: "4px solid #c0392b",
            paddingLeft: 24, margin: 0,
            fontStyle: "italic",
          }}>
            <HighlightedText text={match.sovietText} sharedWords={phase >= 2 ? shared : new Set()} color="#f1c40f" />
          </blockquote>
          <div style={{ fontSize: 12, color: "#666", marginTop: 8 }}>
            — {match.sovietSource}
          </div>
        </div>

        {/* Explanation - appears on scroll */}
        <div style={{
          margin: "40px 0",
          opacity: phase >= 1 ? 1 : 0,
          transform: phase >= 1 ? "translateY(0)" : "translateY(20px)",
          transition: "all 0.8s ease-out 0.2s",
        }}>
          <div style={{
            fontSize: 14, color: "#aaa", lineHeight: 1.7,
            padding: "16px 20px",
            background: "#111122",
            borderRadius: 8,
            borderLeft: "3px solid #f1c40f",
          }}>
            {match.echo_explanation}
          </div>
        </div>

        {/* Modern quote - appears last */}
        <div style={{
          opacity: phase >= 2 ? 1 : 0,
          transform: phase >= 2 ? "translateY(0)" : "translateY(30px)",
          transition: "all 0.8s ease-out",
        }}>
          <div style={{
            fontSize: 11, fontWeight: 700, letterSpacing: 2,
            color: "#3498db", textTransform: "uppercase", marginBottom: 12,
          }}>
            NOW — {match.modernYear} · {yearGap} years later
          </div>
          <blockquote style={{
            fontFamily: "'Georgia', serif",
            fontSize: 22, lineHeight: 1.6, color: "#ddd",
            borderLeft: "4px solid #3498db",
            paddingLeft: 24, margin: 0,
            fontStyle: "italic",
          }}>
            <HighlightedText text={match.modernText} sharedWords={shared} color="#f1c40f" />
          </blockquote>
          <div style={{ fontSize: 12, color: "#666", marginTop: 8 }}>
            — {match.modernSource}
          </div>
        </div>

        {/* Similarity score - appears with modern */}
        <div style={{
          marginTop: 24, textAlign: "center",
          opacity: phase >= 2 ? 1 : 0,
          transition: "all 0.6s ease-out 0.4s",
        }}>
          <span style={{
            fontSize: 13, color: "#888",
            background: "#111122", padding: "6px 16px",
            borderRadius: 20,
          }}>
            {Math.round(match.similarity * 100)}% semantic similarity
          </span>
        </div>
      </div>
    </div>
  );
}

export default function ScrollViz() {
  const [data, setData] = useState(null);
  const [phases, setPhases] = useState({});

  useEffect(() => {
    fetch("/viz_data.json")
      .then(r => r.json())
      .then(d => setData(d))
      .catch(() => {});
  }, []);

  if (!data) return <div style={{ color: "#888", padding: 40 }}>Loading...</div>;

  const matches = data.matches || [];

  return (
    <div style={{
      background: "#0a0a14", color: "#ccc",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }}>
      {/* Hero */}
      <div style={{
        minHeight: "100vh", display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "40px 20px", textAlign: "center",
      }}>
        <h1 style={{
          fontSize: 48, fontWeight: 800, color: "#fff",
          letterSpacing: -1, marginBottom: 16,
        }}>
          I've Seen This Before
        </h1>
        <p style={{
          fontSize: 18, color: "#888", maxWidth: 600, lineHeight: 1.6, marginBottom: 40,
        }}>
          Soviet anti-Zionist propaganda from the 1970s and 1980s is strikingly
          echoed in today's anti-Israel discourse. Scroll to see the parallels.
        </p>
        <div style={{ fontSize: 13, color: "#555", letterSpacing: 2 }}>
          SCROLL ↓
        </div>
      </div>

      {/* Scrollytelling matches */}
      {matches.map((match, idx) => (
        <div key={match.id} style={{ position: "relative" }}>
          {/* Divider */}
          {idx > 0 && (
            <div style={{
              height: 1, background: "linear-gradient(90deg, transparent, #333, transparent)",
              margin: "0 auto", maxWidth: 600,
            }} />
          )}

          <Scrollama
            offset={0.4}
            onStepEnter={({ data }) => {
              setPhases(prev => ({ ...prev, [`${match.id}-${data}`]: true }));
            }}
          >
            <Step data={0}>
              <div style={{ minHeight: "60vh" }}>
                <ScrollMatch
                  match={match}
                  phase={
                    phases[`${match.id}-2`] ? 2 :
                    phases[`${match.id}-1`] ? 1 :
                    phases[`${match.id}-0`] ? 0 : -1
                  }
                />
              </div>
            </Step>
            <Step data={1}>
              <div style={{ minHeight: "40vh" }} />
            </Step>
            <Step data={2}>
              <div style={{ minHeight: "40vh" }} />
            </Step>
          </Scrollama>
        </div>
      ))}

      {/* Closing */}
      <div style={{
        minHeight: "60vh", display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "40px 20px", textAlign: "center",
      }}>
        <p style={{ fontSize: 16, color: "#888", maxWidth: 500, lineHeight: 1.7 }}>
          These are structural parallels in rhetoric — strikingly similar arguments
          made decades apart. Whether the modern speakers learned these framings from
          Soviet sources, or arrived at them independently, the patterns are unmistakable.
        </p>
        <p style={{ fontSize: 13, color: "#555", marginTop: 24 }}>
          Built with NLP/ML · Corpus: {data.soviet_corpus_size} Soviet passages ·
          {data.modern_corpus_size} modern texts analyzed
        </p>
      </div>
    </div>
  );
}
