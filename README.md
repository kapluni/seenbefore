# I've Seen This Before

**Mapping Soviet anti-Zionist propaganda to its modern echoes using NLP and semantic embeddings**

---

> This project was inspired by the work of **[Izabella Tabarovsky](https://www.tabletmag.com/contributors/izabella-tabarovsky)** — senior advisor at the Kennan Institute (Wilson Center) and author of *[Soviet Anti-Zionism and Contemporary Left Antisemitism](https://fathomjournal.org/soviet-anti-zionism-and-contemporary-left-antisemitism/)* and *[Demonization Blueprints](https://doi.org/10.1515/ijfa-2023-0012)*. Her journalism and scholarship first traced the direct lineage from Soviet propaganda to modern anti-Zionist rhetoric, providing the intellectual foundation that this project aims to demonstrate computationally.

---

## What This Is

"I've Seen This Before" uses semantic embeddings to place Soviet anti-Zionist propaganda (1960s–1980s) side by side with modern antisemitic and anti-Zionist rhetoric, surfacing how strikingly — and often verbatim — the language has survived.

The slogans that define today's anti-Zionist movement — "Zionism is racism," "apartheid state," "settler-colonialism" — were not coined by grassroots activists. They were engineered by the Soviet propaganda apparatus, broadcast in 80 languages, and embedded in international institutions where they persist decades after the USSR's collapse.

**Live site**: [iveseenthisbefore.org](https://iveseenthisbefore.org)

## Approach

The system embeds both Soviet propaganda texts and modern rhetoric into a shared vector space using a sentence-transformer model, then identifies passages where the language is strikingly similar — not just topically related, but making the same rhetorical claims with the same framing.

This is **not** a claim of direct causation. Establishing that modern speakers consciously learned these arguments from Soviet sources requires historical evidence beyond NLP. What this project demonstrates is that the rhetorical patterns are strikingly similar — and that the similarity is systematic, not coincidental.

## Trope Taxonomy

The project classifies propaganda into 9 categories, each rooted in documented Soviet rhetorical strategies:

| # | Trope | Description |
|---|-------|-------------|
| 1 | **Zionism = Racism** | Zionism framed as inherently racist; apartheid analogies |
| 2 | **Zionism = Nazism** | Claims of Zionist-Nazi collaboration; Israel compared to Nazi Germany |
| 3 | **Zionism = Imperialism** | Israel as a tool of Western/American imperialism; settler-colonial framing |
| 4 | **Jewish Conspiracy** | Zionist control of media, finance, and governments |
| 5 | **Delegitimization** | Denial of Jewish nationhood or right to self-determination |
| 6 | **Weaponized Antisemitism** | Dismissing antisemitism claims as a "Zionist trick" to silence criticism |
| 7 | **Dual Loyalty** | Jews as agents of a foreign power |
| 8 | **Blood Libel** | Accusations of deliberate child-killing; dehumanization |
| 9 | **Anti-Zionism as Progressive Duty** | Anti-Zionism framed as anti-racist/anti-colonial obligation |

The taxonomy is defined with keyword patterns in [`embedding_pipeline.py` (lines 79–140)](embedding_pipeline.py#L79-L140) under `TROPE_TAXONOMY`. Each trope has weighted keywords matched with word boundary detection to prevent false positives (e.g., "media" inside "immediate").

## Architecture

```
Soviet Corpus (seed passages + Ivanov + 3 archive.org pamphlets)
    ↓
Sentence segmentation → OCR cleanup → Propaganda pre-filter → Trope tagging → Embedding
    ↓
Trope-filtered index (~1,800 passages with propaganda tropes)
    ↓
Modern Corpus (ISCA + CONAN + ADL H.E.A.T. Map + GoldStandard, ~3,900 passages)
    ↓
Embed → Similarity search → Dedup → Diversity enforcement → Quality filter → [LLM verification]
    ↓
viz_data.json → React frontend (static)
/api/analyze → React frontend (live analyzer)
```

## Datasets

### Soviet Corpus (~1,800 filtered passages)

| Source | Year | File |
|--------|------|------|
| Yuri Ivanov, *Caution: Zionism!* | 1970 | [`corpus/soviet_sources/ivanov_caution_zionism_1970_full.txt`](corpus/soviet_sources/ivanov_caution_zionism_1970_full.txt) |
| *Zionism: Instrument of Imperialist Reaction* (Novosti) | 1970 | [`corpus/soviet_sources/novosti_instrument_imperialist_reaction_1970.txt`](corpus/soviet_sources/novosti_instrument_imperialist_reaction_1970.txt) |
| *Anti-Zionist Committee: Aims and Tasks* (Novosti) | 1983 | [`corpus/soviet_sources/anti_zionist_committee_aims_tasks_1983.txt`](corpus/soviet_sources/anti_zionist_committee_aims_tasks_1983.txt) |
| *Zionism: Enemy of Peace and Social Progress* (Progress) | 1985 | [`corpus/soviet_sources/zionism_enemy_peace_progress_1985.txt`](corpus/soviet_sources/zionism_enemy_peace_progress_1985.txt) |
| Anti-Zionist Committee declaration (excerpts) | 1983 | [`corpus/soviet_sources/anti_zionist_committee_declaration_1983.txt`](corpus/soviet_sources/anti_zionist_committee_declaration_1983.txt) |
| Great Soviet Encyclopedia entry on Zionism | — | [`corpus/soviet_sources/great_soviet_encyclopedia_zionism.txt`](corpus/soviet_sources/great_soviet_encyclopedia_zionism.txt) |
| Novick address | 1983 | [`corpus/soviet_sources/novick_anti_zionist_campaign_1983.txt`](corpus/soviet_sources/novick_anti_zionist_campaign_1983.txt) |

The raw Soviet texts contain ~60% noise (historical narrative, quoted Zionist leaders, statistics, citations). The `is_propaganda_chunk()` function in [`embedding_pipeline.py` (line 349)](embedding_pipeline.py#L349) filters these, keeping only passages with genuine propaganda rhetoric. Criteria:

- Rejects chunks that are mostly quoted material (>50% inside quotation marks)
- Rejects citation-heavy passages (3+ reference brackets)
- Rejects pure historical narrative (3+ dates, no propaganda keywords)
- Rejects passages that are mostly proper nouns or statistics (>40% capitalized/numeric)
- **Trope gate**: Only passages with at least one trope classification are used for matching

OCR artifacts from archive.org PDF extraction (hyphenated line breaks, embedded page numbers) are cleaned in `_clean_text()` in `CorpusProcessor` ([`embedding_pipeline.py` line 184](embedding_pipeline.py#L184)).

### Modern Corpus (~3,900 passages)

| Dataset | Size | Content |
|---------|------|---------|
| ISCA GoldStandard 2024 (Zenodo) | 1,838 | IHRA-labeled antisemitic tweets from Indiana University's International Studies of Contemporary Antisemitism |
| ISCA ClassData 2022–2023 | 127 | Multi-group bias data, filtered to Jewish-targeted |
| ISCA HuggingFace | 44 | Additional ISCA multi-group data |
| CONAN Multitarget | 406 | Expert-written hate speech from the CONAN counter-narrative dataset |
| CONAN Dialogues | 547 | Dialogue-format hate speech |
| ADL H.E.A.T. Map | 971 | Incident descriptions from the Anti-Defamation League's Hate, Extremism, Antisemitism, Terrorism map |

Processing is in [`process_modern_sources.py`](process_modern_sources.py), which handles deduplication via normalized text matching, encoding artifact cleanup, and source tagging.

## Embedding Model

**[BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)** — selected via domain-specific benchmarking against 4 candidates.

We tested 10 hand-crafted pairs (5 genuine Soviet→modern echoes + 5 negative controls) across:
- `intfloat/multilingual-e5-large`
- `BAAI/bge-large-en-v1.5`
- `sentence-transformers/all-MiniLM-L6-v2`
- `nomic-ai/nomic-embed-text-v1.5`

BGE-large showed **2x the discrimination gap** between genuine echoes and false positives (0.21 vs 0.11 for E5). Benchmark code is in [`benchmark_models.py`](benchmark_models.py). The embedding engine is in [`embedding_pipeline.py` (line 503)](embedding_pipeline.py#L503) under `EmbeddingEngine`.

Trade-off: BGE-large is English-only (unlike multilingual E5). Since all current corpus texts are English translations, this is acceptable.

## Quality Controls

Embedding similarity alone produces ~50% coherent matches. The pipeline applies multiple layers of filtering:

1. **Near-duplicate dedup** — Normalized text matching catches trivial rephrasing ([`generate_viz_data.py`](generate_viz_data.py))
2. **Soviet passage reuse cap** — Max 2–3 reuses per Soviet passage to prevent one passage dominating
3. **Source diversity enforcement** — No single modern source >40% of match slots
4. **Heuristic quality filter** — `is_weak_match()` in [`generate_viz_data.py`](generate_viz_data.py) removes pro-Israel false positives, too-short texts, and legitimate criticism
5. **LLM echo verification** — Claude Sonnet rates each match as STRONG_ECHO, WEAK_ECHO, NO_ECHO, or FALSE_POSITIVE (`--verify` flag, via `verify_matches_llm()`)
6. **Manual curation** — Final matches are hand-reviewed for genuine rhetorical echoing

### Calibration: Legitimate Criticism

A core design requirement: **legitimate criticism of Israeli policy must score LOW**. The system includes 8 calibration texts representing genuine policy criticism (settlements, humanitarian concerns, peace advocacy, legal arguments). All score below the 0.55 similarity threshold, with an average of **0.43**.

Trope-aware scoring dampens the similarity of texts with no trope overlap by 40%, ensuring "I disagree with settlement policy" doesn't get flagged as propaganda.

### Key Insight: Seed-First Matching

The biggest lesson: **start from the strongest Soviet quotes, then find modern echoes** — not the other way around. All-vs-all matching (embed everything → rank by similarity) produced ~50% incoherent results because cosine similarity measures topical overlap, not rhetorical echoing. Two texts can discuss Israel from opposite angles and still score 0.80+.

The working approach:
1. Start with hand-curated Soviet seed passages (short, punchy, unmistakable propaganda)
2. For each seed, retrieve the top-K closest modern texts
3. Human review selects matches where the **language itself is strikingly similar** — not just the topic, but the specific claim, framing, and rhetorical structure

## File Inventory

```
├── embedding_pipeline.py          # Core ML: corpus processing, embedding, similarity, trope classification
├── generate_viz_data.py           # Bridge to frontend: generates viz_data.json, serves FastAPI analyzer
├── generate_explorer_data.py      # Pre-computes explorer_data.json (passage-level cross-corpus matches)
├── process_modern_sources.py      # Processes 7 modern datasets into unified corpus
├── benchmark_models.py            # Embedding model comparison (BGE vs E5 vs MiniLM vs Nomic)
├── download_datasets.sh           # Downloads ISCA, CONAN, UC Berkeley datasets
├── viz_data.json                  # Generated output consumed by frontend
├── corpus/
│   ├── soviet_sources/            # Soviet propaganda source texts (7 documents)
│   ├── modern_sources/            # Downloaded modern datasets
│   ├── modern_corpus.json         # Processed modern corpus (3,933 passages)
│   └── new_sources/               # Additional datasets (GoldStandard2024, HEATMapData)
├── frontend/                      # Vite + React app
│   ├── src/App.jsx                # Main app (10 tabs: Story, Matches, Explorer, Timeline, Tropes, Methodology, Share, Full History, Background, Thanks)
│   ├── src/useTheme.jsx           # Light/dark theme with system preference detection
│   ├── src/index.css              # CSS variables for theming
│   └── public/
│       ├── background.md          # Condensed background narrative
│       ├── background-comparative.md  # Comparative evidence: Israel singled out
│       └── full-history.md        # Full bibliography (300+ sources)
└── Dockerfile                     # HF Spaces deployment config
```

### Key Classes and Functions

**`embedding_pipeline.py`**:
- `CONFIG` (line 43) — Model name, similarity thresholds, chunk sizes
- `TROPE_TAXONOMY` (line 79) — 9-category taxonomy with weighted keywords
- `Passage` dataclass (line 147) — Core data structure for all corpus passages
- `CorpusProcessor` (line 184) — Text cleaning, chunking, propaganda pre-filtering, trope classification
- `is_propaganda_chunk()` (line 349) — Heuristic filter for Soviet text noise
- `EmbeddingEngine` (line 503) — Sentence-transformer wrapper for batch embedding
- `SimilarityEngine` (line 562) — Cosine similarity search with trope-aware scoring

**`generate_viz_data.py`**:
- `load_soviet_corpus()` — Loads and processes all Soviet source files
- `load_modern_corpus()` — Loads processed modern corpus
- `generate_viz_data()` — Main pipeline: embed, match, filter, output
- `is_weak_match()` — Heuristic quality filter for false positives
- `verify_matches_llm()` — Claude-based echo verification
- `serve_api()` — FastAPI live analyzer endpoint
- `clean_modern_text()` — Encoding artifact and social media cleanup

**`generate_explorer_data.py`**:
- `select_diverse_soviet()` / `select_diverse_modern()` — Diversity-maximizing passage selection
- Outputs pre-computed top-10 cross-corpus matches for interactive exploration

## Running the Project

```bash
# Install Python dependencies (use pyenv virtualenv "seenbefore")
pip install sentence-transformers torch numpy pandas scikit-learn tqdm fastapi uvicorn anthropic

# Process modern corpus (after downloading datasets)
python process_modern_sources.py

# Generate viz_data.json (~5 min on Apple Silicon)
python generate_viz_data.py --generate --max-modern 2000 --top-matches 35

# With LLM verification (requires ANTHROPIC_API_KEY):
ANTHROPIC_API_KEY=sk-ant-... python generate_viz_data.py --generate --verify

# Generate explorer data
python generate_explorer_data.py

# Start frontend dev server
cd frontend && npm run dev   # http://localhost:5173

# Start live analyzer API
python generate_viz_data.py --serve   # http://localhost:8000

# Quick demo (no downloads needed)
python embedding_pipeline.py --demo

# Benchmark embedding models
python embedding_pipeline.py --eval-only
```

## Deployment

- **Frontend**: Cloudflare Pages — `cd frontend && npm run build && npx wrangler pages deploy dist --project-name ive-seen-this-before`
- **API**: Hugging Face Spaces (Docker) — see [`Dockerfile`](Dockerfile)

## Limitations

- **Correlation, not causation**: Semantic similarity shows structural parallels in rhetoric, not a proven chain of transmission. Establishing that modern speakers learned these arguments from Soviet sources requires historical evidence beyond NLP.
- **English only**: All texts are English translations. The BGE-large model is English-only. Russian-language analysis would require switching to a multilingual model.
- **Automated matching precision ~50%**: Embedding similarity captures topical overlap, not rhetorical echoing. Human curation is essential — the displayed matches are hand-reviewed.
- **Corpus bias**: The modern corpus is drawn from academic hate speech datasets and incident reports, which over-represent extreme rhetoric. Campus discourse and mainstream media are underrepresented.
- **Threshold sensitivity**: Confidence tiers (high ≥0.85, medium ≥0.70, low ≥0.55) are calibrated for BGE-large's score distribution and may need adjustment for other models.

## Tech Stack

- **Embeddings**: [sentence-transformers](https://www.sbert.net/) with [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **Backend**: Python 3.14, FastAPI, NumPy
- **Frontend**: Vite + React, Recharts, React Markdown, React Scrollama, html-to-image
- **LLM**: Claude API (Sonnet) for trope classification and match verification
- **Deployment**: Cloudflare Pages (frontend), Hugging Face Spaces (API)

## Acknowledgments

This project builds on the scholarship of **Izabella Tabarovsky**, **Robert Wistrich**, **Jeffrey Herf**, and **Ion Mihai Pacepa**, who documented how Soviet propaganda shaped modern anti-Zionist discourse. Datasets were provided by **ISCA** (Indiana University), **ADL**, and the **CONAN** project. Much of the research, writing, and code was developed with **Claude** (Anthropic).

## License

This project is intended for educational and research purposes. The Soviet texts are public domain or freely available translations. Modern corpus data is used under the terms of their respective academic licenses.

## Author

**Ilya Kaplun** — ML Engineering Manager. Emigrated from the USSR in 1991. With a ton of help from [Anthropic Claude](https://claude.ai).
