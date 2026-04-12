# Analysis Improvement Plan

Research conducted April 2026. Prioritized by impact-to-effort ratio.

---

## Phase 1: High Impact, Low Effort (days)

### 1. NLI Cross-Encoder Filtering
**Branch:** `experiment/nli-filtering`
**Problem:** Cosine similarity captures topic but not argument direction. Two texts about Israel from opposite perspectives can score 0.80+.
**Solution:** Use `cross-encoder/nli-deberta-v3-base` to classify each (soviet, modern) candidate pair as entailment/contradiction/neutral. Filter out contradictions.
**Expected impact:** Push automated precision from ~50% to ~70-80%.
**Implementation:** ~50 lines in `generate_viz_data.py` or a new `nli_filter.py`.

### 2. Cross-Encoder Reranking
**Branch:** `experiment/cross-encoder-reranking`
**Problem:** Bi-encoder retrieval ranks by topical overlap, not rhetorical similarity.
**Solution:** After bi-encoder retrieves top-50 candidates per Soviet passage, use `BAAI/bge-reranker-v2-m3` or `cross-encoder/ms-marco-MiniLM-L-6-v2` to jointly read both texts and rerank. Take top-5 after reranking.
**Expected impact:** Documented precision@1 improvements from 0.75 to 1.00.

### 3. Statistical Quality Metrics
**Branch:** `experiment/statistical-metrics`
**Problem:** No aggregate statistical evidence that matches are better than chance.
**Metrics to implement:**
- Permutation test (shuffle Soviet/modern labels 10K times, compare real vs null distribution)
- Cohen's d effect size (curated matches vs random pairs)
- Bootstrap 95% CI on mean match similarity
- ROC/AUC (19 positive matches + random negatives)
- Mann-Whitney U test (match scores vs legitimate criticism scores)
**Implementation:** ~100 lines of numpy/scipy code. Uses existing 19 matches + 8 calibration texts.

---

## Phase 2: High Impact, Moderate Effort (1-2 weeks)

### 4. Instruction-Tuned Embeddings
**Branch:** `experiment/instruction-embeddings`
**Problem:** BGE-large-en-v1.5 has no task awareness — embeds all text the same way regardless of downstream intent.
**Candidates:**
- `BAAI/bge-en-icl` — In-context learning: provide 3-5 example pairs, model adapts embedding space on the fly
- `intfloat/e5-mistral-7b-instruct` — Prepend task instruction describing what "rhetorical echo" means
- `Alibaba-NLP/gte-Qwen2-7B-instruct` — Instruction-tuned, 70.24 MTEB
**Evaluation:** Benchmark on existing 10 domain-specific test pairs (5 genuine echoes + 5 negatives). Measure gap between good and bad pairs.

### 5. LLM Claim Extraction
**Branch:** `experiment/claim-extraction`
**Problem:** Matching full passages conflates multiple claims. A passage may contain 3 claims, only 1 of which echoes.
**Solution:** Use Claude to decompose each passage into atomic claims ("Zionism serves Western imperialism", "Zionists control media"), then match claims rather than passages.
**Expected impact:** More precise matching at the claim level; better explanations of what specifically echoes.

### 6. Propaganda Technique Detection
**Branch:** `experiment/propaganda-techniques`
**Problem:** Trope taxonomy captures *what* is said, not *how* it's said.
**Solution:** Run `QCRI/PropagandaTechniquesAnalysis-en-BERT` (18 techniques: loaded language, appeal to fear, name-calling, etc.) on both corpora. Two texts using same trope AND same technique = stronger match.
**Sources:** SemEval-2023 Task 3, SemEval-2024 Task 4.

---

## Phase 3: Longer-Term

### 7. Fine-Tune Sentence Transformer
Use 19 curated matches as positives + rejected candidates as hard negatives. Contrastive learning with `CosineSimilarityLoss` or `ContrastiveLoss` via sentence-transformers library.

### 8. Multi-Signal Ensemble Scoring
Combine: embedding similarity + NLI entailment probability + trope overlap + propaganda technique overlap → weighted composite score or learned classifier.

### 9. Diachronic Word Embeddings
Track how terms like "colonialism," "apartheid," "racism" shifted meaning between Soviet-era and modern usage. Based on Hamilton et al. (ACL 2016) methods.

---

## Key Models Referenced

| Model | Type | MTEB | Notes |
|-------|------|------|-------|
| `BAAI/bge-large-en-v1.5` | Bi-encoder | 63-64 | Current model |
| `BAAI/bge-en-icl` | Bi-encoder + ICL | 71.24 | In-context learning for embeddings |
| `intfloat/e5-mistral-7b-instruct` | Bi-encoder + instruction | ~70 | Task-description-aware |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker | — | Pairs with BGE embeddings |
| `cross-encoder/nli-deberta-v3-base` | NLI cross-encoder | — | Entailment/contradiction/neutral |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker | — | Fast general reranker |
| `QCRI/PropagandaTechniquesAnalysis-en-BERT` | Propaganda detector | — | 18 techniques, fragment-level |
| `Alibaba/Qwen3-Embedding-8B` | Bi-encoder | 70.58 | Best open model, needs GPU |

## Key Papers

- Hamilton et al., "Diachronic Word Embeddings" (ACL 2016)
- "Decoding Persuasion" survey (Frontiers in Communication, 2024)
- "LLMs in Argument Mining" survey (arXiv 2506.16383, 2025)
- "Claim Extraction for Fact-Checking" (arXiv 2502.04955, 2025)
- "A Survey on LLM-as-a-Judge" (arXiv 2411.15594, 2024)
- SemEval-2023 Task 3: propaganda.math.unipd.it/semeval2023task3/
