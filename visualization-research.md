# Visualization Research: "I've Seen This Before"

Research into visualization techniques, libraries, and design patterns for presenting Soviet-to-modern propaganda lineage. Organized by visualization category with concrete implementation recommendations.

---

## 1. Current State Assessment

### What Exists Today
- **6-tab React app** (Matches, Analyzer, Timeline, Tropes, Calibration, Social Cards)
- **Recharts** as the only charting library (horizontal bar chart for trope distribution)
- **Inline CSS** dark theme with red (Soviet) / blue (modern) color coding
- **Static match cards**: 3-column layout (Soviet text | similarity donut | Modern text)
- **Conic gradient donuts** for similarity scores
- **Linear timeline** with alternating left/right events
- **Social card generator** with THEN/NOW split design

### What's Missing (per architecture doc)
- Geographic mapping of incidents
- Network graphs showing trope propagation
- Filtering/search across matches
- Temporal heatmaps (trope surges over time)
- Embedding space visualization
- Scrollytelling/narrative walkthrough
- Text diff/highlight showing specific rhetorical parallels
- Oral history video integration

---

## 2. High-Impact Visualization Ideas

### 2.1 Trope Propagation Network Graph

**Concept**: An interactive force-directed graph where Soviet source documents are nodes on the left, modern texts are nodes on the right, and edges represent detected echoes. Nodes are colored by trope, sized by number of connections. Clicking a node highlights all its echoes.

**Why it matters**: This is the single most powerful way to show the *systemic* nature of the rhetorical transfer — it's not one-off coincidences, it's a pattern.

**Libraries**:
| Library | Pros | Cons | React Support |
|---------|------|------|---------------|
| **@react-sigma/core** (Sigma.js) | Fast WebGL rendering, handles 10K+ nodes, great for large graphs | Steeper learning curve | First-class React bindings |
| **react-force-graph** (by vasturiano) | Simple API, 2D/3D/VR modes, good defaults | Less customizable | React wrapper |
| **D3-force + React** | Maximum control, well-documented | Manual React integration, SVG perf limits at scale | DIY |
| **vis.js Network** | Easy to use, good clustering | Heavier bundle, less React-native | Community wrapper |
| **Cytoscape.js** | Academic-grade, good for research viz | Complex API | react-cytoscapejs |

**Recommendation**: `react-force-graph-2d` for quick prototyping (simple API, good defaults). Migrate to `@react-sigma/core` if performance matters at scale.

**Data shape**: Already available — each match in `viz_data.json` connects a Soviet source to a modern text with trope labels and similarity scores.

**Layout idea**: Bipartite layout with Soviet nodes pinned to the left column, modern nodes on the right, force simulation on the edges. Color edges by trope. Hovering a trope in the legend highlights all edges of that trope.

---

### 2.2 Sankey / Alluvial Diagram: Rhetoric Flow

**Concept**: A Sankey diagram showing how Soviet propaganda tropes flow into modern rhetoric categories. Left column: Soviet source documents. Middle column: 9 trope categories. Right column: modern source types (campus, social media, political discourse). Band width = number of matches.

**Why it matters**: Immediately communicates *volume* and *direction* of rhetorical transfer. A viewer sees at a glance that ZIONISM_IMPERIALISM flows heavily from Ivanov → campus BDS resolutions.

**Libraries**:
- **D3-sankey** + React: Maximum control, well-documented
- **recharts** (already installed): Has a basic `<Sankey>` component — check if sufficient
- **nivo** (`@nivo/sankey`): Beautiful defaults, React-native, interactive, responsive
- **Google Charts Sankey**: Quick but less customizable

**Recommendation**: Try `@nivo/sankey` first — it has the best out-of-box aesthetics for React and supports tooltips, link/node coloring, and responsive containers. Falls back to d3-sankey if more control is needed.

**Data transformation**: Group matches by (soviet_source → trope → modern_source_type) and count.

---

### 2.3 Embedding Space Scatter Plot (t-SNE / UMAP)

**Concept**: 2D scatter plot of all Soviet and modern passages projected from the 1024-dim BGE-large embedding space into 2D via UMAP. Soviet passages as red dots, modern as blue. Connected pairs have lines drawn between them. Clusters of similar rhetoric become visually apparent.

**Why it matters**: Makes the abstract concept of "semantic similarity" tangible. Users can see that certain modern texts literally cluster with Soviet propaganda in meaning-space.

**Implementation**:
1. **Backend**: Run UMAP reduction on all embeddings (Python `umap-learn` library), output 2D coordinates to `viz_data.json`
2. **Frontend**: Render with one of:

| Library | Best For | Notes |
|---------|----------|-------|
| **regl-scatterplot** | 100K+ points, WebGL | Fastest, minimal API |
| **deck.gl ScatterplotLayer** | Geospatial + scatter, WebGL | Overkill unless also doing maps |
| **@nivo/scatterplot** | <5K points, beautiful | SVG-based, slower at scale |
| **Plotly.js** | Quick interactive plots | Heavier bundle but easy |
| **Observable Plot** | Concise, modern API | Newer, less React integration |

**Recommendation**: With ~5,700 total passages (1,800 Soviet + 3,900 modern), `regl-scatterplot` or Plotly.js both work. Plotly is faster to implement; regl is faster to render.

**Interaction**: Hover shows passage text. Click shows the full match card. Lasso select to explore a cluster. Toggle trope colors.

---

### 2.4 Highlighted Text Diff / Rhetorical Parallels

**Concept**: Instead of showing Soviet and modern text as plain blocks, highlight the specific phrases, framings, and word choices that make them echoes. Think "code diff" but for rhetoric.

**Why it matters**: This is the "aha moment" — when a user sees the *exact* words and argument structures highlighted side-by-side, the echo becomes undeniable.

**Approaches**:

1. **Keyword/phrase highlighting**: Use the trope taxonomy keywords to highlight matching terms in both texts. Simple, fast, no ML needed.

2. **Sentence-level alignment**: For each sentence in the modern text, find the most similar sentence in the Soviet text (using embeddings). Draw connecting arcs between aligned sentences. Libraries: custom SVG arcs.

3. **LLM-generated annotations**: Use Claude to identify and annotate the specific parallel phrases. Already partially implemented via `echo_explanation` — extend to produce structured highlights:
   ```json
   {
     "sovietHighlights": [{"start": 12, "end": 45, "label": "dehumanization frame"}],
     "modernHighlights": [{"start": 0, "end": 33, "label": "dehumanization frame"}],
     "connections": [{"soviet": 0, "modern": 0}]
   }
   ```

4. **Word-level attention visualization**: Show which words in the modern text are most responsible for the high similarity score (using embedding gradient analysis or simpler TF-IDF overlap).

**Libraries**: No special library needed — React `<span>` elements with highlight colors + SVG connection lines. For fancier rendering: `react-diff-viewer` (adapted) or custom.

**5. N-gram overlap highlighting**: Extract all 2-grams and 3-grams from both texts, highlight any that appear in both (or are semantically equivalent via embeddings). Shows shared phrases like "tool of imperialism", "weaponize antisemitism" directly. Library: `jsdiff` (github.com/kpdecker/jsdiff) provides word-level diffing primitives.

**Recommendation**: Start with approach 1 (keyword highlighting using trope taxonomy). Upgrade to approach 5 (n-gram overlap) for richer highlighting. Use approach 3 (LLM annotations) for curated top matches. This gives the highest impact for the investment.

---

### 2.5 Scrollytelling / Guided Narrative

**Concept**: A scroll-driven experience that walks the user through the evidence step by step:
1. Open with a powerful modern quote
2. Scroll → reveal the Soviet source saying almost the same thing
3. Scroll → zoom out to show this is one of dozens of matches
4. Scroll → show the trope network
5. Scroll → show the timeline of how this rhetoric traveled
6. End with calibration: "Here's what legitimate criticism looks like — it scores differently"

**Why it matters**: This is how you convert casual visitors into understanding the project's thesis. The Matches tab is an *explorer* for people already engaged; scrollytelling is the *front door*.

**Libraries**:
| Library | Approach | Notes |
|---------|----------|-------|
| **Scrollama** | Intersection Observer-based scroll triggers | Lightweight, battle-tested (NYT, The Pudding) |
| **react-scrollama** | React wrapper for Scrollama | Drop-in for React apps |
| **GSAP ScrollTrigger** | Animation-driven scroll | More control, heavier |
| **Framer Motion + scroll** | React-native animations | Already popular in React ecosystem |
| **react-scroll-parallax** | Parallax effects | Good for visual polish |

**Recommendation**: `react-scrollama` for scroll step triggers + `framer-motion` for animations. This is the standard stack for React scrollytelling.

**Design reference**: The Pudding's visual essays (e.g., "Film Dialogue" piece), NYT's "Snowfall", The Washington Post's interactives.

---

### 2.6 Geographic Heatmap of Modern Incidents

**Concept**: Map showing where modern antisemitic incidents (from ADL H.E.A.T. Map data) occur, colored by which Soviet trope they echo. Cluster by campus, city, or state.

**Why it matters**: Makes the phenomenon *geographically real* — users can see it's happening at their university or in their city.

**Libraries**:
- **react-map-gl** (Mapbox/MapLibre wrapper): Best for React, interactive, supports heatmaps and clusters
- **Leaflet + react-leaflet**: Free, open-source, good enough for incident dots
- **deck.gl HexagonLayer**: Beautiful heatmap aggregation, WebGL

**Data available**: ADL H.E.A.T. Map has 3,672 incidents with location data. 971 are Israel/Zionism-related.

**Recommendation**: `react-leaflet` for MVP (free, no API key). Upgrade to `react-map-gl` for production polish.

---

### 2.7 Temporal Heatmap / Trope Surge Detection

**Concept**: A heatmap grid where X-axis = time (years or months), Y-axis = trope category, color intensity = frequency. Shows when specific propaganda themes surge.

**Why it matters**: Reveals that certain events (Oct 7, Durban Conference, campus cycles) trigger specific trope activations, mirroring how the Soviets activated propaganda around geopolitical events.

**Libraries**:
- **@nivo/heatmap**: Beautiful, responsive, React-native
- **recharts** (existing): Can hack with a colored grid
- **D3 calendar heatmap**: Classic GitHub-contribution-style

**Data needed**: Modern corpus passages need timestamps (many from Twitter datasets have dates). Group by (month, trope) and count.

---

### 2.8 "Propaganda Playbook" Process Diagram

**Concept**: Instead of showing the 9 tropes as a flat list, arrange them as a **sequential propaganda process** showing how Soviet rhetoric built its case — and how modern rhetoric follows the same playbook:
1. Delegitimize Jewish nationhood → 2. Equate Zionism with racism/Nazism → 3. Frame as imperialist tool → 4. Invoke conspiracy theories → 5. Dismiss antisemitism claims → 6. Frame anti-Zionism as progressive duty

**Why it matters**: Shows that these aren't random talking points but a coordinated rhetorical strategy. Inspired by Propwatch.org's organization of propaganda by psychological process rather than flat taxonomy.

**Implementation**: Horizontal flowchart with numbered steps, each expandable to show Soviet examples (left) and modern examples (right). Could use a simple CSS grid or a lightweight flow library like `reactflow`.

---

### 2.9 "Echo Explorer" (Inspired by Yale Intertext)

**Concept**: An interactive view where the full Soviet corpus text is displayed with segments color-coded by how frequently they have modern echoes. Click any highlighted segment to see its closest modern matches in a side panel.

**Why it matters**: Lets users explore the *source material* directly and discover echoes organically, rather than only seeing pre-curated match pairs. Based on Yale DHLab's Intertext project (MIT-licensed React components, directly adaptable).

**Implementation**: Render Soviet texts with `<span>` elements colored by match density (grey = no matches, yellow = 1-2 matches, red = 3+ matches). Side panel shows ranked modern matches for the selected segment.

---

## 3. Architectural Recommendations

### Quick Wins (< 1 day each)
1. **Add keyword highlighting to match cards** — highlight trope taxonomy terms in Soviet and modern text blocks using colored `<span>` elements. Uses existing data, no new libraries.
2. **Add filtering to Matches tab** — filter by trope, confidence level, source, year range. Pure React state, no new library.
3. **Upgrade trope chart** — add the Sankey view as an alternative to the current bar chart. Try `@nivo/sankey`.

### Medium Effort (1-3 days each)
4. **Network graph tab** — new tab showing the trope propagation network. Use `react-force-graph-2d`.
5. **Scrollytelling landing page** — guided narrative using `react-scrollama` as the project's front door.
6. **Text diff highlights** — LLM-generated highlight annotations for top 20 curated matches.

### Larger Investments (1+ week each)
7. **Embedding scatter plot** — requires backend UMAP computation, new frontend rendering.
8. **Geographic map** — requires geocoding ADL H.E.A.T. Map data, map tile integration.
9. **Temporal heatmap** — requires timestamp extraction and normalization across all modern sources.

### Recommended Library Additions
```json
{
  "@nivo/sankey": "^0.87.0",
  "react-force-graph-2d": "^1.25.0",
  "react-scrollama": "^2.3.0",
  "framer-motion": "^11.0.0",
  "react-leaflet": "^4.2.0"
}
```

---

## 4. Design Principles for This Project

### Credibility First
- Every visualization should include methodology notes
- Show confidence levels and thresholds transparently
- Always include the calibration counterexample (legitimate criticism scores low)
- Avoid sensationalism in color/animation choices — let the data speak

### The "Aha Moment" Hierarchy
1. **Immediate**: Side-by-side text with highlighted parallel phrases (text diff)
2. **Systemic**: Network graph showing it's not one coincidence but a pattern
3. **Historical**: Timeline/Sankey showing the rhetorical pipeline across decades
4. **Personal**: Geographic map showing it's happening near the viewer

### Accessibility
- All charts need color-blind-safe palettes (current trope colors need review)
- Text alternatives for all visual data
- Keyboard navigation for interactive elements
- Mobile-responsive (current inline CSS approach is fragile for mobile)

### Shareability
- Every visualization state should be URL-addressable (deep linking)
- Export individual charts as images for social media
- The social card generator should support all new viz types, not just match cards

---

## 5. Inspiration Projects & References

### Digital Humanities Text Comparison
- **Yale DHLab Intertext** (github.com/YaleDHLab/intertext) — MIT-licensed React app for detecting and visualizing text reuse. Three modes: Search (find modern passages reusing Soviet language), Compare (track one passage across corpus), Visualize (color-code by reuse frequency). **Directly adaptable** — its React visualization components could be integrated into our Vite app. [Demo](https://duhaime.s3.amazonaws.com/yale-dh-lab/intertext/demo/index.html)
- **Tesserae** (DHQ paper: digitalhumanities.org/dhq/vol/16/1/000602/000602.html) — Intertext detection in Latin/Greek literature via bigram matching. Their shared-phrase highlighting approach could inspire a "shared phrase highlighter" for Soviet/modern text pairs.
- **Voyant Tools** (voyant-tools.org) — Text analysis dashboard with word clouds, trends, correlations. Good model for multi-panel text analysis.
- **Quantitative Intertextuality Survey** (arxiv.org/html/2510.27045v1) — Recent survey covering Text-PAIR, passim, TRACER, and deep neural approaches. Confirms our BGE-large pipeline is aligned with current scholarly methods.

### Propaganda & Disinformation Visualization
- **Hamilton 68 / Hamilton 2.0** (securingdemocracy.gmfus.org/hamilton-dashboard/) — Alliance for Securing Democracy dashboard tracking Russian influence. Key pattern: trending topics bar charts + timeline of messaging theme intensity. Directly inspires a "Trope Intensity Timeline" showing how tropes surge around events.
- **Propwatch** (propwatch.org) — Catalogs propaganda techniques organized by psychological vulnerability exploited, cross-referenced with media. Their taxonomy-as-process approach inspires arranging our 9 tropes as a **propaganda playbook sequence** rather than a flat list.
- **Disinformation Observatory** (disinfobs.com/index.php/narrative-dashboard/) — AI-driven narrative dashboard with heat maps tracking disinfo trends. Good UX reference for our temporal heatmap.
- **EUvsDisinfo** (euvsdisinfo.eu) — Database of disinformation cases with search, filtering, and trend visualization.
- **Stanford Internet Observatory** reports — Annotated network graphs, temporal charts, and narrative walkthroughs.

### Scrollytelling & Narrative Data Viz
- **The Pudding** (pudding.cool) — Gold standard for scroll-driven data essays. Key insight from their process: "start with a single data point, then zoom out to the pattern." Their [resources page](https://pudding.cool/resources/) documents their approach.
- **D3.js + Scrollama + React tutorial** (itnext.io, Nov 2025) — Step-by-step guide for combining D3 + react-scrollama with sticky graphics in React. Exactly our stack.
- **Scrollama** (github.com/russellsamora/scrollama) — Created by Russell Samora of The Pudding. Uses IntersectionObserver, no scroll event listeners. [Introduction post](https://pudding.cool/process/introducing-scrollama/).
- **Reuters Graphics** — Clean, methodical scroll-driven explanations of complex topics.

### Network & Graph Visualization
- **Sigma.js + @react-sigma** (sigmajs.org) — WebGL graph rendering with first-class React bindings. Architecture: Graphology (data) → Sigma (rendering) → @react-sigma (React). [Practical guide](https://www.menudo.com/react-sigma-js-the-practical-guide-to-interactive-graph-visualization-in-react/).
- **Cosmograph** (cosmograph.app) — Handles hundreds of thousands of nodes in-browser. Free for non-commercial use (CC BY-NC 4.0).
- **D3 force-directed graphs** — [Observable example](https://observablehq.com/@d3/force-directed-graph-component), [React+D3+TypeScript guide](https://medium.com/@qdangdo/visualizing-connections-a-guide-to-react-d3-force-graphs-typescript-74b7af728c90).

### Embedding Space Visualization
- **regl-scatterplot** (github.com/flekschas/regl-scatterplot) — WebGL scatter plot, handles up to 20M points. Supports lasso selection, zoom, pan. Best fit for our ~5,700 passages. [Demo](https://flekschas.github.io/regl-scatterplot/).
- **Deepscatter** (github.com/nomic-ai/deepscatter) — Nomic's library, scales to billions of points using tiled data. Runs fully static (no server needed).
- **TensorFlow Embedding Projector** (projector.tensorflow.org) — Interactive 3D UMAP/t-SNE. Quick prototyping via TSV upload.
- **Nomic Atlas** (atlas.nomic.ai) — Cloud-hosted, automatic clustering and topic labeling. Free tier: 250K points. Good for development exploration.
- **WizMap** (arxiv.org/html/2306.09328) — Research on scalable embedding visualization.

---

## 6. Priority Recommendation

If I had to pick **three visualizations** to build next, in order:

1. **Highlighted text parallels** (Section 2.4) — Highest impact per effort. Transforms match cards from "interesting" to "undeniable." Start with keyword highlighting, upgrade to LLM annotations.

2. **Scrollytelling landing page** (Section 2.5) — Essential for public launch. Without a guided narrative, only already-engaged visitors will explore the tabs. This is the conversion funnel.

3. **Trope propagation network** (Section 2.1) — The "systemic" view that transforms this from "here are some similar quotes" to "here is a pattern of rhetorical transfer." Most shareable visualization for academic and media audiences.

These three cover the "aha moment hierarchy": immediate recognition (text highlights), guided understanding (scrollytelling), and systemic evidence (network graph).
