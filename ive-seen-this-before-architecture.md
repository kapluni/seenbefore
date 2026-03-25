# "I've Seen This Before" — Technical Architecture Plan

## Project Summary

A semantic similarity engine and public-facing platform that maps Soviet anti-Zionist propaganda (1960s–1980s) to modern antisemitic and anti-Zionist rhetoric on campuses, social media, and in public discourse. The goal is to make the lineage of today's anti-Zionist language visible and undeniable.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Soviet Corpus │    │ Modern Corpus│    │ Oral History     │   │
│  │ (historical)  │    │ (ongoing)    │    │ Archive (future) │   │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘   │
│         │                   │                     │             │
│         ▼                   ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Document Store (PostgreSQL + pgvector)      │    │
│  │              + Object Storage (S3) for raw documents     │    │
│  └──────────────────────────┬──────────────────────────────┘    │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│                     PROCESSING LAYER                            │
│                              │                                  │
│  ┌───────────────────────────▼──────────────────────────────┐   │
│  │                  Embedding Pipeline                       │   │
│  │  OCR/Digitization → Chunking → Translation → Embedding   │   │
│  └───────────────────────────┬──────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────▼──────────────────────────────┐   │
│  │                Similarity Engine                          │   │
│  │  Trope Taxonomy → Semantic Search → Match Scoring        │   │
│  │  → Provenance Linking → Confidence Calibration           │   │
│  └───────────────────────────┬──────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────▼──────────────────────────────┐   │
│  │               Trend & Analytics Engine                    │   │
│  │  Time-series tracking → Spike detection → Trope          │   │
│  │  frequency analysis → Platform comparison                │   │
│  └───────────────────────────┬──────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│                     APPLICATION LAYER                           │
│                              │                                  │
│  ┌────────────┐  ┌──────────▼───┐  ┌────────────┐              │
│  │ Public Site │  │ Analyzer API │  │ Campaign   │              │
│  │ (explore,   │  │ (paste text, │  │ Generator  │              │
│  │  browse,    │  │  get Soviet  │  │ (social    │              │
│  │  learn)     │  │  matches)    │  │  content)  │              │
│  └────────────┘  └──────────────┘  └────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Soviet Corpus Assembly (Months 1–3)

This is the foundation. No model is useful without a well-curated historical corpus.

### 1.1 Source Acquisition

| Source | Format | Language | Acquisition Method |
|--------|--------|----------|-------------------|
| Yuri Ivanov, "Beware: Zionism" (1969) | Book | Russian (some English translations exist) | Library scan / Internet Archive / academic request |
| Kichko, "Judaism Without Embellishment" (1963) | Book | Russian/Ukrainian | Digitized copies via academic libraries |
| Novosti Press Agency pamphlets | Pamphlets | English, Russian, Arabic, French | AJHS archive, Library of Congress, British Library |
| Anti-Zionist Committee of the Soviet Public declarations | Official documents | Russian | Marxists.org archive, academic collections |
| Yevseyev, "Fascism Under a Blue Sky" | Book | Russian | Academic libraries |
| CPSU official definitions of Zionism | Encyclopedia entries, party documents | Russian | Great Soviet Encyclopedia (digitized), party congress proceedings |
| Soviet newspaper articles (Pravda, Izvestia, Komsomolskaya Pravda) | Newspaper | Russian | East View databases, library microfilm |
| UN Resolution 3379 and surrounding debate transcripts | Official UN docs | English, Russian, Arabic, French | UN digital archive (freely available) |
| Soviet political cartoons | Images | Visual + Russian captions | Academic collections, museum archives |
| Mahmoud Abbas's 1982 Moscow dissertation | Academic thesis | Russian/Arabic | Referenced in multiple academic sources |
| Soviet-era films and documentaries on Zionism | Video/transcripts | Russian | Academic film archives |

### 1.2 Processing Pipeline

```
Raw Document (scan/PDF/image/text)
    │
    ├─→ [If image/scan] OCR via Tesseract (Russian model) or Google Cloud Vision
    │
    ├─→ [If Russian] Machine translation (Google Translate API / DeepL)
    │       + human review for key passages (recruit Russian-speaking volunteers)
    │
    ├─→ Manual quality check against existing English translations where available
    │
    ▼
Clean bilingual text (Russian original + English translation)
    │
    ├─→ Sentence-level segmentation
    │
    ├─→ Named entity extraction (people, orgs, countries)
    │
    ├─→ Trope/theme tagging (manual + LLM-assisted, see taxonomy below)
    │
    ▼
Structured document in corpus DB
    │
    ├─→ Metadata: source, date, author, publication, target audience, language
    │
    ├─→ Embeddings: sentence-level and passage-level (see 2.1)
    │
    └─→ Trope labels: which propaganda themes are present
```

### 1.3 Trope Taxonomy (v1)

Based on the scholarly literature (Tabarovsky, ADL, Wistrich), define a structured taxonomy of Soviet anti-Zionist propaganda themes. This is the Rosetta Stone of the project.

```
TROPE TAXONOMY v1
─────────────────

1. ZIONISM = RACISM
   - Zionism as inherently racist ideology
   - "Zionism is a form of racism and racial discrimination" (UN 3379 language)
   - Comparison of Zionism to apartheid

2. ZIONISM = NAZISM/FASCISM
   - Zionist-Nazi collaboration allegations
   - Israel employs "Nazi methods"
   - IDF compared to Wehrmacht/SS
   - Gaza compared to concentration camps/ghettos

3. ZIONISM = IMPERIALISM/COLONIALISM
   - Israel as tool of Western/American imperialism
   - Settler-colonial framing
   - Zionism as enemy of national liberation movements
   - Israel allied with global capitalist exploitation

4. JEWISH/ZIONIST CONSPIRACY
   - Zionist control of media
   - Zionist control of finance/banking
   - Zionist lobby controlling governments
   - Hidden Zionist influence ("the Zionist entity")
   - Protocols-derived narratives repackaged

5. DELEGITIMIZATION OF JEWISH SELF-DETERMINATION
   - Denial of Jewish nationhood
   - Zionism as "artificial" movement
   - Jewish connection to Israel as fabricated
   - "Zionist entity" (denying statehood)

6. WEAPONIZATION OF ANTISEMITISM
   - Claims that antisemitism accusations are "Zionist tricks"
   - Dismissal of Jewish victimhood
   - "Crying antisemitism to deflect criticism"

7. DUAL LOYALTY
   - Jews as agents of a foreign power
   - Conflation of diaspora Jews with Israeli state
   - Accusations of disloyalty to home country

8. BLOOD LIBEL / ATROCITY PROPAGANDA
   - Accusations of deliberate child killing
   - Claims of genocidal intent
   - Dehumanization of Israelis/Zionists

9. ANTI-ZIONISM AS PROGRESSIVE DUTY
   - Framing opposition to Zionism as anti-racist
   - Zionism as obstacle to world peace
   - Solidarity with Palestinians as leftist/progressive imperative
   - Boycott as moral obligation
```

**Annotation approach:** Each passage in the Soviet corpus gets tagged with one or more tropes. Start with manual annotation by domain experts (recruit from academic contacts, Tabarovsky's network), then train a classifier to assist with scaling.

---

## Phase 2: Modern Corpus & Embedding Infrastructure (Months 2–4)

### 2.1 Modern Content Sources

| Source | Content Type | Access Method | Update Frequency |
|--------|-------------|---------------|-----------------|
| CyberWell open database | Vetted antisemitic social media posts | app.cyberwell.org (register for access) | Ongoing |
| ISCA Twitter dataset | 6,941 labeled tweets | Zenodo (CC 4.0) | Static (2019-2021) |
| ADL H.E.A.T. Map | Incident descriptions | CSV download | Monthly |
| Campus BDS resolutions | Resolution text | Manual collection from SJP/BDS sites | As published |
| Protest chant databases | Text transcriptions | Manual collection from video transcriptions | As available |
| Social media (X, TikTok, Instagram) | Posts, comments | CyberWell partnership or custom collection via platform APIs / Bright Data | Ongoing |
| Academic papers / op-eds | Long-form text | Web scraping, manual curation | Ongoing |
| UN/NGO resolutions and reports | Official documents | UN digital archive, NGO websites | As published |

### 2.2 Embedding Strategy

**Model selection:** Use a multilingual sentence transformer that handles both Russian and English well.

```
Recommended models (evaluate all three):

1. multilingual-e5-large (Microsoft)
   - Strong multilingual performance
   - Good for semantic similarity tasks
   - 1024-dim embeddings

2. LaBSE (Language-agnostic BERT Sentence Embeddings)
   - Specifically designed for cross-lingual similarity
   - Handles Russian-English pairs well
   - 768-dim embeddings

3. BGE-M3 (BAAI)
   - State-of-the-art multilingual retrieval
   - Supports dense, sparse, and multi-vector retrieval
   - Good for both short and long passages
```

**Embedding granularity:**

```
For each document, generate embeddings at three levels:

1. Sentence-level: Individual sentences → fine-grained matching
   ("Zionism is a weapon of imperialism" ↔ "Zionism is settler-colonialism")

2. Passage-level: 3-5 sentence windows → contextual matching
   (A paragraph describing Zionist-Nazi collaboration ↔ a campus speech
    comparing Gaza to the Holocaust)

3. Document-level: Full document → thematic similarity
   (A complete Novosti pamphlet ↔ a full BDS resolution)
```

### 2.3 Vector Storage

```
PostgreSQL + pgvector
├── soviet_passages (id, text_ru, text_en, source_metadata, trope_labels, embedding)
├── modern_passages (id, text, source_metadata, trope_labels, embedding, timestamp)
├── matches (soviet_id, modern_id, similarity_score, trope_overlap, human_verified)
└── tropes (id, name, description, examples, soviet_count, modern_count)

Why pgvector over a dedicated vector DB:
- Simpler ops for a small team / volunteer project
- Can do hybrid queries (filter by trope + vector similarity)
- Good enough performance at this scale (tens of thousands, not billions)
- Familiar tooling (standard SQL)
```

---

## Phase 3: Similarity Engine (Months 3–5)

### 3.1 Matching Pipeline

```
Input: Modern text (social media post, speech excerpt, resolution text, etc.)
    │
    ├─→ Preprocessing
    │     - Language detection
    │     - Translation if needed
    │     - Sentence segmentation
    │
    ├─→ Trope Classification
    │     - LLM-based classifier assigns trope labels from taxonomy
    │     - Used for filtering and explainability
    │
    ├─→ Embedding Generation
    │     - Sentence and passage-level embeddings
    │
    ├─→ Semantic Search (pgvector ANN search)
    │     - Find top-k most similar Soviet passages
    │     - Filter by trope overlap for precision
    │
    ├─→ Match Scoring
    │     - Cosine similarity (base score)
    │     - Trope overlap bonus (same trope category = higher relevance)
    │     - Source authority weight (official CPSU doc > random newspaper)
    │     - Temporal proximity penalty (prefer older Soviet sources for impact)
    │
    ├─→ LLM Explanation Generation
    │     - For each top match, generate a human-readable explanation
    │     - "This post echoes Soviet propaganda theme X, first used in Y context"
    │     - Include historical context snippet
    │
    ▼
Output: Ranked list of Soviet matches with:
    - Original Soviet text (Russian + English)
    - Source citation and date
    - Similarity score and explanation
    - Trope classification
    - Historical context
```

### 3.2 Match Quality and Calibration

This is critical — false matches undermine credibility.

```
Confidence tiers:

HIGH CONFIDENCE (auto-publish)
- Cosine similarity > 0.85 AND trope overlap AND human-verified source pair
- Example: Direct quote from Soviet encyclopedia definition of Zionism
  matched against campus resolution using near-identical language

MEDIUM CONFIDENCE (human review queue)
- Cosine similarity 0.70-0.85 OR trope overlap without high similarity
- Example: Thematic similarity but different specific language

LOW CONFIDENCE (research only, not public-facing)
- Cosine similarity 0.55-0.70, potentially coincidental overlap
- Useful for trend analysis, not individual claims

REJECTED
- Below threshold or flagged by human reviewer as spurious
```

**Human-in-the-loop:** Build a simple review UI where volunteer domain experts (historians, Russian speakers, antisemitism researchers) can verify/reject matches and provide corrections. This improves the system over time and ensures academic credibility.

### 3.3 Anti-Gaming and Fairness Considerations

- The tool should NOT be used to label all criticism of Israel as Soviet-derived. Build in explicit guardrails:
  - Legitimate policy criticism (e.g., "I disagree with settlement expansion") should score LOW on the similarity engine
  - The taxonomy distinguishes between criticism of Israeli policy and demonization/delegitimization
  - Include a "not a match" explanation pathway: "This text criticizes Israeli policy but does not employ propaganda tropes"
- Transparency: all methodology, taxonomy, and training data should be publicly documented
- Adversarial testing: have both pro-Israel and critical-of-Israel reviewers test the system for bias before launch

---

## Phase 4: Application Layer (Months 4–7)

### 4.1 Public Website — "I've Seen This Before"

**Core experience: The Explorer**

```
Landing page:
  "The language of today's anti-Zionism was written by Soviet propagandists
   decades ago. Explore the connections."

Main sections:

1. BROWSE BY TROPE
   - Select a trope category (e.g., "Zionism = Racism")
   - See Soviet-era examples on the left, modern examples on the right
   - Timeline showing when the trope was created, how it spread, where it appears today
   - Key documents and quotes

2. ANALYZE TEXT
   - Paste any text (social media post, article excerpt, resolution)
   - Get back: matched Soviet sources, trope classification, historical context
   - Shareable result card for social media

3. TIMELINE
   - Interactive timeline from 1948 → present
   - Key Soviet propaganda milestones (Six-Day War response, UN 3379, Anti-Zionist Committee)
   - Modern echoes mapped to their historical origins
   - Can filter by trope

4. PERSONAL STORIES (future: Oral History Archive integration)
   - Video/audio testimonials from Soviet Jewish immigrants
   - "I heard this in Moscow in 1975. Now I hear it on American campuses."
   - Searchable by theme, era, country of origin

5. RESOURCES
   - For educators: lesson plans (partner with Refusenik Project / Lookstein Center)
   - For students: quick-reference cards for responding to common anti-Zionist claims
   - For researchers: full methodology, downloadable data, API documentation
```

**Tech stack (keep it simple for a volunteer project):**

```
Frontend: Next.js (React) + Tailwind
  - Static generation for browse/timeline pages (fast, cheap hosting)
  - Client-side API calls for the analyzer tool

Backend: Python (FastAPI)
  - Embedding generation endpoint
  - Similarity search endpoint
  - LLM explanation endpoint

Database: PostgreSQL + pgvector (Supabase or Railway for managed hosting)

LLM: Claude API (Sonnet for classification + explanation generation)

Hosting: Vercel (frontend) + Railway or Fly.io (backend)

Estimated monthly cost: $50-150 (mostly LLM API calls)
```

### 4.2 Analyzer API

Public API for other organizations to integrate:

```
POST /api/analyze
{
  "text": "Zionism is a settler-colonial project...",
  "language": "en",        // auto-detected if omitted
  "detail_level": "full"   // "summary" | "full" | "research"
}

Response:
{
  "tropes_detected": [
    {
      "trope": "ZIONISM_IMPERIALISM_COLONIALISM",
      "confidence": 0.91,
      "explanation": "The phrase 'settler-colonial project' maps directly to..."
    }
  ],
  "soviet_matches": [
    {
      "text_en": "Serving as the front squad of colonialism and neo-colonialism...",
      "text_ru": "Выступая в роли передового отряда колониализма и неоколониализма...",
      "source": "CPSU Anti-Zionist Committee Declaration, 1983",
      "similarity_score": 0.88,
      "trope": "ZIONISM_IMPERIALISM_COLONIALISM",
      "context": "This language was part of the official Soviet campaign to..."
    }
  ],
  "is_legitimate_criticism": false,
  "summary": "This text employs language that originated in Soviet anti-Zionist propaganda..."
}
```

### 4.3 Social Media Campaign Generator

For the "I've Seen This Before" social campaign, auto-generate shareable content:

```
Input: A high-confidence match pair (Soviet source + modern echo)

Output: Split-screen image card
  ┌─────────────────────┬─────────────────────┐
  │     THEN (1975)     │     NOW (2024)       │
  │                     │                      │
  │  "Zionism is a      │  "Zionism is a       │
  │   weapon of         │   settler-colonial   │
  │   imperialism"      │   project"           │
  │                     │                      │
  │  — Novosti Press    │  — [campus source]   │
  │    Agency, Moscow   │                      │
  │                     │                      │
  │  #IveSeenThisBefore │                      │
  └─────────────────────┴─────────────────────┘
```

---

## Phase 5: Scaling & Partnerships (Months 6+)

### 5.1 Data Partnerships to Pursue

| Partner | What They Provide | What We Provide |
|---------|-------------------|-----------------|
| CyberWell | Real-time antisemitic content feed, platform relationships | Soviet-origin analysis layer for their existing data |
| ADL | Incident data, educational reach, credibility | Technical tool they can embed/reference |
| Tabarovsky / ISGAP | Academic validation, Soviet source expertise, network | Technical infrastructure for her qualitative research |
| Lookstein Center / Refusenik Project | Educational distribution channel, existing lesson plans | Interactive digital tool to complement their curriculum |
| StandWithUs / Hillel | Campus distribution, student user base | Debate prep data, analyzer tool |

### 5.2 Oral History Integration (Future)

When the oral history archive component is ready:

```
Interview Pipeline:
  Record (video/audio)
    → Transcribe (Whisper API, handles Russian-accented English well)
    → Translate (if Russian-language interview)
    → Segment into topic-tagged clips
    → Embed and index for semantic search
    → Link to relevant Soviet corpus entries and modern matches

User experience:
  "Show me immigrants talking about [dual loyalty accusations]"
  → Returns video clips of immigrants describing their experiences
  → Alongside Soviet propaganda examples of the same trope
  → Alongside modern examples of the same trope
```

---

## Team & Roles Needed

| Role | Skills | Phase | Commitment |
|------|--------|-------|------------|
| **Project Lead (Ilya)** | ML engineering, project management | All | Ongoing |
| **Russian-speaking research assistant (1-2)** | Russian fluency, Soviet history knowledge | 1-2 | 10-15 hrs/week for 3 months |
| **Frontend developer (1)** | React/Next.js, data visualization | 4 | 10 hrs/week for 2 months |
| **Domain expert advisor** | Antisemitism scholarship (Tabarovsky?) | All | Advisory, 2-4 hrs/month |
| **Annotation volunteers (3-5)** | Reading comprehension, attention to detail | 1-3 | 5 hrs/week for 2 months |
| **Historian advisor** | Soviet history, Cold War propaganda | 1-2 | Advisory, 2-4 hrs/month |

**Where to find volunteers:**
- CJP Hineni Network (skills-based matching)
- Tech Professionals for Israel network (if you build it)
- Boston-area university students (Brandeis, BU, Harvard — Judaic studies departments)
- Hack the Hate hackathon participants (if you organize one)

---

## MVP Milestones

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| **M1: Corpus v1** | Month 2 | 500+ annotated Soviet passages from 5-10 key sources, trope-tagged |
| **M2: Embedding pipeline** | Month 3 | Working similarity search: paste text → get Soviet matches |
| **M3: Internal demo** | Month 4 | Working analyzer with top-50 curated match pairs, shareable to advisors |
| **M4: Public beta** | Month 5 | Website with Browse by Trope + Analyzer tool, 100+ verified match pairs |
| **M5: Campaign launch** | Month 6 | Social media campaign with first batch of "Then/Now" cards |
| **M6: API + partnerships** | Month 7 | Public API, integrations with 1-2 partner organizations |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Credibility attack** ("this is just hasbara") | Undermines entire project | Academic advisors, transparent methodology, open-source data, include legitimate criticism examples that score LOW |
| **False matches** embarrass the project | Loss of trust | Conservative confidence thresholds, human review for all public-facing matches, adversarial testing |
| **Corpus too small** to be useful | Weak results | Start with the most well-documented sources; quality over quantity. 500 well-annotated passages > 5,000 sloppy ones |
| **Legal/copyright issues** with Soviet texts | Takedown risk | Most Soviet-era texts are not under Western copyright; consult with IP attorney for edge cases. Academic fair use applies for research |
| **Scope creep** | Never ships | Strict MVP: corpus + similarity search + basic web UI. Everything else is Phase 2+ |
| **Volunteer burnout** | Stalls progress | Keep initial team small (3-5), well-scoped tasks, visible progress. Ilya's EM skills are key here |
| **Platform TOS issues** for social media collection | Data access cut off | Partner with CyberWell/FOA who already have platform relationships rather than scraping independently |

---

## Budget Estimate (MVP through Month 7)

| Item | Cost |
|------|------|
| Cloud hosting (DB, backend, frontend) | $100-150/month |
| LLM API costs (Claude for classification/explanation) | $50-100/month |
| Translation API (DeepL/Google) | $50/month during corpus building |
| Domain + CDN | $20/month |
| **Total monthly (steady state)** | **~$200-300/month** |
| **Total through MVP (7 months)** | **~$1,500-2,000** |

Note: This assumes volunteer labor for development and annotation. If paying for any contract work (e.g., Russian translator for quality review), budget accordingly.

---

## Getting Started: First Two Weeks

1. **Set up the project repo** (GitHub, public or private initially)
2. **Acquire 3-5 key Soviet source texts** in English translation:
   - Start with the Novosti pamphlet analyzed in the Quillette article
   - CPSU Anti-Zionist Committee declaration (available on Marxists.org)
   - UN 3379 debate transcripts (freely available)
   - Excerpts from "Beware: Zionism" (check Internet Archive)
3. **Draft the trope taxonomy v1** and get feedback from 1-2 domain experts
4. **Set up pgvector** locally and embed a small test set (50 Soviet passages + 50 modern examples from CyberWell/ISCA dataset)
5. **Build a minimal similarity search** notebook — prove the concept works
6. **Reach out to Tabarovsky** with the concept and early results

---

---

## Appendix: Corpus Acquisition Status (as of March 2026)

### Acquired — Seed Corpus Files

The following files are saved in `/corpus/soviet_sources/`:

| File | Words | Content |
|------|-------|---------|
| `anti_zionist_committee_declaration_1983.txt` | 288 | Full Pravda declaration + SAPIR Journal translation |
| `great_soviet_encyclopedia_zionism.txt` | 228 | Official CPSU definitions of Zionism |
| `ivanov_caution_zionism_1970_excerpts.txt` | 730 | Key propaganda passages + term glossary |
| `novick_anti_zionist_campaign_1983.txt` | 448 | Key passages + list of Soviet anti-Zionist writers |

### Immediate Downloads Available (full texts)

These are freely available and should be downloaded to expand the corpus:

1. **Ivanov "Caution: Zionism!" FULL TEXT (~50,000 words)**
   - PDF: https://www.marxists.org/subject/jewish/caution-zionism.pdf
   - Internet Archive: https://archive.org/details/yuri-ivanov-caution-zionism-progress-1970
   - LA Museum of the Holocaust: http://www.lamoth.info/index.php?p=digitallibrary/digitalcontent&id=5872
   - `pdftotext caution-zionism.pdf` to extract
   - **This single document covers ALL 9 tropes in the taxonomy**

2. **Novick address FULL TEXT**
   - https://www.marxists.org/subject/jewish/novick-anti-zionist.htm
   - Public domain (Creative Commons)

3. **Anti-Zionist Committee press conference (June 1983)**
   - https://archive.org/details/anti-zionist-committee-of-soviet-public-opinion-aims-and-tasks

4. **UN Resolution 3379 debate transcripts**
   - https://undocs.org/en/A/RES/3379(XXX)

5. **Tabarovsky "Demonization Blueprints" (academic paper with extensive primary source quotes)**
   - https://www.degruyterbrill.com/document/doi/10.26613/jca/5.1.97/pdf

### Secondary Source Extraction Targets

These articles contain translated/quoted Soviet propaganda passages that can be extracted:

| Source | URL | Key Content |
|--------|-----|-------------|
| "Zombie Anti-Zionism" (Tabarovsky, Tablet) | tabletmag.com | Novosti pamphlet analysis, New Delhi conference docs |
| "The Language of Soviet Propaganda" (Quillette) | quillette.com | Detailed analysis of 76-page Novosti pamphlet with word counts |
| "Red Terror" (AIIA) | internationalaffairs.org.au | Pipeline from Soviet propaganda to campus discourse |
| "The Anti-Zionist Lexicon" (Jones, ToI) | blogs.timesofisrael.com | KGB operations, Abbas dissertation, Arafat/PLO connections |
| ADL "Contemporary Anti-Zionism's Connections..." | adl.org | Zionist-Nazi trope origins, Abbas dissertation quotes |
| Soviet anti-Zionism Wikipedia article | en.wikipedia.org | Extensive documented quotes from primary sources |
| "Anti-Zionist Committees of the American Public" (SAPIR) | sapirjournal.org | Full Anti-Zionist Committee declaration translation |
| Tabarovsky Forward op-ed | forward.com | Personal testimony + propaganda examples |

### Key Academic Contacts for Deeper Access

| Person | Affiliation | What They May Provide |
|--------|------------|----------------------|
| Izabella Tabarovsky | Kennan Institute/Wilson Center, ISGAP | Personal research corpus, Soviet source access, academic validation |
| Gunther Jikeli | Indiana University ISCA | ISCA Twitter dataset methodology, annotation guidelines |
| Sasha Zborovsky | University of Pennsylvania (PhD candidate) | Access to KGB documents, Dutch archives on Soviet Jewish emigration |

### Modern Corpus Sources (parallel track)

| Source | Status | Action |
|--------|--------|--------|
| CyberWell open database | Register at app.cyberwell.org | Request researcher access |
| ISCA Zenodo dataset (6,941 tweets) | Download from doi.org/10.5281/zenodo.7932888 | CC 4.0 license |
| ADL H.E.A.T. Map CSV | Download from adl.org heat map page | Monthly updates |
| UC Berkeley hate speech dataset | Download from HuggingFace | 39,565 comments |
| CONAN counter-narratives | Download from github.com/marcoguerini/CONAN | Includes Jewish-targeted pairs |

### Estimated Corpus Size After Phase 1 (Month 2)

| Category | Estimated Passages | Source |
|----------|-------------------|--------|
| Soviet primary sources | 200-300 | Ivanov full text + Committee declarations + encyclopedia |
| Soviet secondary extraction | 100-200 | Quotes from scholarly articles listed above |
| Modern antisemitic content | 200-300 | CyberWell + ISCA dataset + manually collected |
| **Total seed corpus** | **500-800 passages** | |

This is sufficient for a compelling MVP demo and proof-of-concept similarity search.

---

*Architecture document created: March 2026*
*Author: Ilya*
*Status: Draft — ready for review and refinement*
