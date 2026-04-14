# Protocols of the Elders of Zion — Experiment Findings

**Date:** 2026-04-13 to 2026-04-14
**Pipeline:** `generate_viz_data.py --generate --max-modern 1000 --top-matches 35 --enrich --include-historical`
**Model:** BAAI/bge-large-en-v1.5
**Claim extraction:** Claude Opus (claude-opus-4-20250514)
**Technique detection:** facebook/bart-large-mnli (zero-shot, SemEval techniques)

## Overview

This experiment adds the *Protocols of the Elders of Zion* (first published in Russian c.1903; text used is the 1920 English 2nd edition published by "The Britons", London) as a **historical antisemitism corpus** to test whether modern antisemitic rhetoric echoes pre-Soviet as well as Soviet-era propaganda.

## Corpus Statistics

| Corpus | Raw passages | After propaganda filter | With tropes |
|--------|-------------|------------------------|-------------|
| Soviet (Ivanov + 4 pamphlets) | ~4600 | ~2575 | 1964 |
| Protocols (filtered, no publisher prefaces) | 845 | 300 | 154 |
| Modern (6 datasets, sampled) | — | — | 1000 |

The Protocols file required filtering: the 1920 edition includes publisher prefaces, a "Prefatory Note to the Second Edition," Nilus's introduction, and back-matter advertisements for other antisemitic publications. Only the actual Protocols text (lines 294–4252 of the source file) was used, skipping ~293 lines of editorial material and ~100 lines of ads/table of contents.

## Enrichment Pipeline

This experiment introduced a three-signal ensemble scoring system:

1. **Claim similarity (weight: 60%)** — Claude Opus extracts atomic claims from each passage, embeds them with BGE-large, and computes max claim-pair cosine similarity. This measures whether two texts make the *same argument*, not just discuss the same topic.

2. **Technique overlap (weight: 40%)** — A zero-shot classifier (BART-large-MNLI) detects SemEval propaganda techniques (loaded language, appeal to fear, name-calling, etc.) in each passage. Jaccard overlap measures shared rhetorical methods.

3. **Claim floor gate** — When claim similarity < 0.65, the weighting shifts to 75%/25% to prevent technique overlap from inflating scores for topically-related but argumentatively-different texts.

### Technical Issues Solved
- **MPS memory contention**: BGE-large (~1.3GB) and BART (~1.6GB) cannot coexist on MPS; BART runs on CPU
- **Opus safety refusals**: Academic context framing in the prompt prevents safety refusals on hateful content extraction
- **JSON parsing**: Opus sometimes wraps responses in ```json fences; added fallback parsing

## Final Results

### Soviet→Modern (31 matches)
| Metric | Value |
|--------|-------|
| Matches | 31 (1 high, 29 medium, 1 low) |
| Avg ensemble score | ~0.68 |
| Top ensemble | 0.843 (claim=0.953, tech=0.900) |
| Lowest | 0.511 (claim=0.515, tech=0.500) |

### Protocols→Modern (32 matches, 3 filtered below 0.45)
| Metric | Value |
|--------|-------|
| Matches after filtering | 32 |
| Avg ensemble score | 0.570 |
| Top ensemble | 0.675 (claim=0.680, tech=0.667) |

### Protocols→Soviet (19 matches, 1 filtered below 0.45)
| Metric | Value |
|--------|-------|
| Matches after filtering | 19 |
| Avg ensemble score | 0.595 |
| Top ensemble | 0.761 (claim=0.713, tech=0.833) |

### Calibration (unchanged)
- Legitimate criticism avg: **0.43** (all 8 examples below 0.55)
- One outlier: "Budget Discussion" at 0.67 (has trope keyword match — needs investigation)

## Assessment: Matches Are Not Compelling

While the pipeline produces matches with reasonable scores, **manual review of the Protocols→Modern and Protocols→Soviet matches did not reveal the kind of striking, obvious rhetorical echoes** that make the Soviet→Modern matches compelling.

### Why the matches are weak

1. **The Protocols is too stylistically different.** It uses 19th-century conspiratorial language ("learned elders," "goyim," "our scheme") that doesn't map well to either Soviet academic-sounding propaganda or modern Twitter rhetoric. The cosine similarity catches topical overlap (both discuss Jewish power) but the *rhetorical register* is completely different.

2. **Claim similarity is moderate, not high.** The best Protocols→Modern claim similarity is 0.680 (vs. 0.953 for the best Soviet→Modern match). The claims extracted from the Protocols are too general ("Jews seek world domination") to produce tight matches with specific modern claims.

3. **The three-layer lineage is theoretically interesting but hard to demonstrate with NLP alone.** Showing that Text A is similar to Text B, and Text B is similar to Text C, doesn't prove A→B→C transmission. The Protocols→Soviet similarity could just reflect shared antisemitic themes rather than actual rhetorical inheritance.

4. **The editorial preface problem.** Even after filtering, some passages from the 1920 English edition contain translator/editor commentary rather than the original forgery text. These introduce anachronisms (e.g., references to "Bolsheviki" that couldn't appear in a 1903 text).

### What DID work

1. **The ensemble scoring pipeline is sound.** Claim extraction via Opus + technique detection via BART produces more meaningful scores than cosine similarity alone. The claim floor gate correctly demotes matches where only technique overlap inflates the score.

2. **The claim extraction approach is validated.** 148 cached claim entries, with only 6 empty (texts too short/fragmentary). Opus handles hateful content extraction well with academic framing.

3. **The Protocols→Soviet similarity is highest** (avg 0.595 vs. 0.570 for Protocols→Modern), consistent with the thesis that Soviet propaganda adapted Protocols-era tropes. But the gap is too small to be conclusive.

## Learnings for the Main Project

1. **Ensemble scoring should be applied to Soviet→Modern matches.** The claim similarity + technique overlap approach produces better quality signals than cosine alone. The top Soviet→Modern match (ensemble 0.843, claim 0.953) is genuinely compelling.

2. **The Protocols experiment is worth mentioning in the methodology** as evidence that the project considered deeper historical roots, but should not be a primary feature of the visualization — the matches don't stand on their own.

3. **Claim floor gate is effective.** Matches where claim similarity < 0.65 but technique overlap is high are correctly demoted. This prevents "same propaganda style, different argument" matches from ranking highly.

4. **BART on CPU is the right architecture.** Running technique detection on CPU avoids MPS memory contention with the embedding model. Performance is acceptable (~3-5 sec per passage).

5. **The year attribution matters.** The Protocols source file is a 1920 English translation, not the 1903 Russian original. Displaying "1903/1920" with both dates avoids anachronisms.

## Recommendations

1. **Keep the enrichment pipeline** (claim extraction + technique detection + ensemble scoring) and apply it to Soviet→Modern matches on main
2. **Archive the Historical Roots tab** — it's technically functional but the matches aren't compelling enough for public display
3. **Keep the Protocols corpus and code** for future research — a more curated approach (hand-picking the strongest Protocols passages) might yield better results
4. **Investigate the "Budget Discussion" calibration outlier** (scoring 0.67 with trope match)
5. **Consider adding Kichko's "Judaism Without Embellishment" (1963)** to the Soviet corpus — it's the earliest and most explicitly antisemitic Soviet text and might bridge the Protocols→Soviet gap more convincingly
