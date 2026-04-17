"""
I've Seen This Before — Embedding Pipeline
==========================================

A complete pipeline for:
1. Processing the Soviet corpus into embeddable chunks
2. Evaluating multilingual embedding models
3. Building a similarity search index
4. Testing against modern antisemitic/anti-Zionist content
5. Generating match reports

Prerequisites:
    pip install sentence-transformers torch numpy pandas scikit-learn tqdm

Optional (for full pipeline):
    pip install psycopg2-binary pgvector  # if using PostgreSQL
    pip install anthropic                  # for LLM-based trope classification

Usage:
    python embedding_pipeline.py              # Run full pipeline
    python embedding_pipeline.py --eval-only  # Just evaluate models
    python embedding_pipeline.py --demo       # Quick demo with sample data
"""

import os
import json
import re
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "corpus_dir": "./corpus/soviet_sources",
    "modern_dir": "./corpus/modern_sources",
    "output_dir": "./output",
    "index_dir": "./index",

    # Embedding models to evaluate (in order of preference)
    "models": [
        "intfloat/multilingual-e5-large",     # Best overall multilingual
        "sentence-transformers/LaBSE",         # Best for cross-lingual pairs
        "BAAI/bge-m3",                         # SOTA multilingual retrieval
        "sentence-transformers/all-MiniLM-L6-v2",  # Fast baseline (English only)
    ],

    # Default model after evaluation
    "default_model": "BAAI/bge-large-en-v1.5",

    # Chunking parameters
    "chunk_max_words": 60,        # Max words per chunk (was 100; shorter = sharper propaganda claims)
    "chunk_overlap_words": 10,    # Overlap between chunks
    "min_chunk_words": 25,        # Skip chunks shorter than this

    # Similarity thresholds
    "high_confidence": 0.85,
    "medium_confidence": 0.70,
    "low_confidence": 0.55,

    # Match parameters
    "top_k": 10,                  # Number of matches to return
}


# ============================================================
# TROPE TAXONOMY
# ============================================================

TROPE_TAXONOMY = {
    "ZIONISM_RACISM": {
        "name": "Zionism = Racism",
        "description": "Zionism as inherently racist ideology; apartheid comparisons",
        "keywords": ["racist", "racism", "apartheid", "racial intolerance", 
                     "chauvinism", "racial discrimination", "segregation",
                     "ethnic cleansing", "supremacy", "supremacist"],
    },
    "ZIONISM_NAZISM": {
        "name": "Zionism = Nazism/Fascism",
        "description": "Zionist-Nazi collaboration; Israel employs Nazi methods",
        "keywords": ["nazi", "fascist", "fascism", "hitler", "genocide",
                     "holocaust", "concentration camp", "ghetto", "blitzkrieg",
                     "extermination", "gas chamber", "ethnic cleansing"],
    },
    "ZIONISM_IMPERIALISM": {
        "name": "Zionism = Imperialism/Colonialism",
        "description": "Israel as tool of Western imperialism; settler-colonial framing",
        "keywords": ["imperialism", "imperialist", "colonial", "colonialism",
                     "settler", "occupation", "annexation", "neo-colonial",
                     "liberation movement", "anti-colonial", "oppressor"],
    },
    "JEWISH_CONSPIRACY": {
        "name": "Jewish/Zionist Conspiracy",
        "description": "Zionist control of media, finance, governments",
        "keywords": ["control", "lobby", "influence", "media", "finance",
                     "banking", "conspiracy", "secret", "hidden hand",
                     "puppet", "manipulation", "propaganda service"],
    },
    "DELEGITIMIZATION": {
        "name": "Delegitimization of Jewish Self-Determination",
        "description": "Denial of Jewish nationhood; 'Zionist entity'",
        "keywords": ["entity", "artificial", "illegitimate", "no right to exist",
                     "fabricated", "invented", "settler state", "not a country"],
    },
    "WEAPONIZED_ANTISEMITISM": {
        "name": "Weaponization of Antisemitism Claims",
        "description": "Dismissing antisemitism accusations as Zionist tricks; tokenizing anti-Zionist Jews",
        "keywords": ["anti-semitic acts", "playing the victim", "weaponize",
                     "deflect", "silence criticism", "smear", "hasbara",
                     "absurd are attempts", "present criticizing them",
                     "anti-zionist not anti", "jews also oppose",
                     "jewish voices", "jews against zionism"],
    },
    "DUAL_LOYALTY": {
        "name": "Dual Loyalty",
        "description": "Jews as agents of foreign power; conflation with Israel",
        "keywords": ["loyalty", "allegiance", "agent", "fifth column", "traitor",
                     "foreign", "espionage", "spy", "infiltrate"],
    },
    "BLOOD_LIBEL": {
        "name": "Blood Libel / Atrocity Propaganda",
        "description": "Accusations of deliberate child killing; dehumanization",
        "keywords": ["child killer", "baby", "blood", "murder", "slaughter",
                     "massacre", "butcher", "disembowel", "savage"],
    },
    "ANTI_ZIONISM_PROGRESSIVE": {
        "name": "Anti-Zionism as Progressive Duty",
        "description": "Opposition to Zionism framed as anti-racist/progressive; 'good Jews' reject Zionism",
        "keywords": ["progressive", "solidarity", "boycott", "divest", "sanctions",
                     "liberation", "justice", "resistance", "struggle",
                     "anti-communism", "reactionary",
                     "not all jews", "don't have to be", "good jews", "jews can be good",
                     "forced to support", "not in my name", "as a jew",
                     "don't represent", "doesn't represent", "do not represent"],
    },
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Passage:
    """A single passage from the corpus."""
    id: str
    text: str
    source: str                          # e.g., "ivanov_caution_zionism_1970"
    source_title: str                    # e.g., "Caution: Zionism!"
    author: str
    year: int
    language: str                        # "en" or "ru"
    corpus: str                          # "soviet" or "modern"
    trope_labels: list = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d


@dataclass
class Match:
    """A match between a Soviet passage and a modern passage."""
    soviet_passage: Passage
    modern_passage: Passage
    similarity_score: float
    trope_overlap: list
    confidence_tier: str                 # "high", "medium", "low"
    explanation: str = ""


# ============================================================
# CORPUS PROCESSING
# ============================================================

class CorpusProcessor:
    """Process raw text files into embeddable passages."""

    def __init__(self, config=CONFIG):
        self.config = config
        # Load system dictionary for OCR period-split correction
        self._dictionary = set()
        dict_path = Path("/usr/share/dict/words")
        if dict_path.exists():
            self._dictionary = {w.strip().lower() for w in dict_path.read_text().splitlines() if len(w.strip()) >= 2}

    def chunk_text(self, text: str, source_meta: dict) -> list[Passage]:
        """Split text into overlapping chunks suitable for embedding."""
        # Clean the text
        text = self._clean_text(text)

        # Split into sentences (simple approach — upgrade to spaCy for production)
        sentences = self._split_sentences(text)

        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            if word_count < 3:  # Skip very short fragments
                continue

            if current_word_count + word_count > self.config["chunk_max_words"] and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= self.config["min_chunk_words"]:
                    chunks.append(chunk_text)

                # Keep overlap
                overlap_words = 0
                overlap_start = len(current_chunk)
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_words += len(current_chunk[i].split())
                    if overlap_words >= self.config["chunk_overlap_words"]:
                        overlap_start = i
                        break
                current_chunk = current_chunk[overlap_start:]
                current_word_count = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_word_count += word_count

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= self.config["min_chunk_words"]:
                chunks.append(chunk_text)

        # Create Passage objects
        passages = []
        for i, chunk in enumerate(chunks):
            passage_id = f"{source_meta['source']}_{i:04d}"
            passages.append(Passage(
                id=passage_id,
                text=chunk,
                source=source_meta.get("source", "unknown"),
                source_title=source_meta.get("title", "Unknown"),
                author=source_meta.get("author", "Unknown"),
                year=source_meta.get("year", 0),
                language=source_meta.get("language", "en"),
                corpus=source_meta.get("corpus", "soviet"),
                metadata=source_meta,
            ))

        return passages

    def classify_tropes_keyword(self, passage: Passage) -> list[str]:
        """Simple keyword-based trope classification.
        Use as baseline; upgrade to LLM-based for production."""
        text_lower = passage.text.lower()
        detected = []

        for trope_id, trope_info in TROPE_TAXONOMY.items():
            for keyword in trope_info["keywords"]:
                # Use word boundary matching to avoid false positives
                # (e.g., "media" matching inside "immediate")
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    detected.append(trope_id)
                    break  # One match per trope is enough

        return detected

    def classify_tropes_llm(self, passage: Passage, client=None) -> list[str]:
        """LLM-based trope classification using Claude API.
        More accurate than keyword matching, especially for coded language."""
        if client is None:
            try:
                from anthropic import Anthropic
                client = Anthropic()
            except ImportError:
                print("anthropic package not installed, falling back to keyword classification")
                return self.classify_tropes_keyword(passage)

        trope_descriptions = "\n".join([
            f"- {tid}: {tinfo['name']} — {tinfo['description']}"
            for tid, tinfo in TROPE_TAXONOMY.items()
        ])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Classify this text passage according to which Soviet anti-Zionist propaganda tropes it contains.

TROPE CATEGORIES:
{trope_descriptions}

TEXT:
"{passage.text}"

Return ONLY a JSON array of matching trope IDs. If none match, return [].
Example: ["ZIONISM_RACISM", "ZIONISM_IMPERIALISM"]"""
            }]
        )

        try:
            # Extract JSON from response
            text = response.content[0].text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return self.classify_tropes_keyword(passage)

    def _clean_text(self, text: str) -> str:
        """Remove comments, headers, metadata from corpus files."""
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            # Skip comment lines and headers
            if line.strip().startswith("#"):
                continue
            if line.strip().startswith("##"):
                continue
            # Skip empty lines
            if not line.strip():
                continue
            # Remove quote marks used for extracted quotes
            line = line.strip().strip('"').strip()
            if line:
                clean_lines.append(line)

        # Fix OCR period-split words across lines BEFORE joining:
        # "So.\n viet" → "Soviet", "Zi.\nonism" → "Zionism"
        # The 1985 pamphlet OCR uses period instead of hyphen at line breaks
        fixed_lines = []
        i = 0
        while i < len(clean_lines):
            line = clean_lines[i]
            if (i + 1 < len(clean_lines) and
                re.search(r'(?<![.A-Z])[A-Z][a-z]+\.\s*$', line) and
                clean_lines[i + 1] and clean_lines[i + 1][0].islower()):
                # Remove the trailing period and join with next line
                line = re.sub(r'([A-Z][a-z]*)\.\s*$', r'\1', line)
                line = line + clean_lines[i + 1]
                i += 2
            else:
                i += 1
            fixed_lines.append(line)
        clean_lines = fixed_lines

        text = " ".join(clean_lines)

        # Fix inline period-split OCR artifacts: "Com.mittee" → "Committee"
        # The 1985 pamphlet uses periods where hyphens should be at line breaks
        def _is_real_word(word):
            """Check if word (or its base form) is in the dictionary."""
            w = word.lower()
            if w in self._dictionary:
                # Also verify each fragment alone isn't a word (avoid "in.reality" → "inreality")
                return True
            # Try stripping common suffixes to find base form
            for suffix in ('s', 'ed', 'ing', 'er', 'ers', 'es', 'ly', 'ment',
                           'ness', 'tion', 'sion', 'ous', 'ive', 'ity', 'ful',
                           'less', 'able', 'ible', 'ment', 'ence', 'ance'):
                if w.endswith(suffix):
                    stem = w[:-len(suffix)]
                    if stem in self._dictionary:
                        return True
                    # Handle e-dropping: "provoked" → "provok" → "provoke"
                    if (stem + 'e') in self._dictionary:
                        return True
            # Handle doubled consonant + suffix: "letters" → "letter"
            if len(w) > 4 and w[-1] == w[-2] and w[:-1] in self._dictionary:
                return True
            return False

        # Function words that are almost always standalone (not OCR fragments)
        _function_words = {'a', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he',
                           'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on',
                           'or', 'so', 'to', 'up', 'us', 'we', 'am', 'are', 'the',
                           'for', 'and', 'but', 'not', 'you', 'all', 'can', 'had',
                           'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how',
                           'its', 'may', 'new', 'now', 'old', 'see', 'way', 'who',
                           'did', 'get', 'say', 'she', 'too', 'use'}

        def fix_period_split(m):
            before = m.group(1)
            after = m.group(2)
            joined = before + after
            joined_is_word = _is_real_word(joined)
            before_is_word = before.lower() in self._dictionary
            after_is_word = after.lower() in self._dictionary

            # If joined form isn't a real word, keep the period
            if not joined_is_word:
                return m.group(0)
            # If the before-fragment is a function word, it's standalone → keep period
            if before.lower() in _function_words:
                return m.group(0)
            # If before is >= 4 chars and both fragments are real words → sentence boundary
            if len(before) >= 4 and before_is_word and after_is_word:
                return m.group(0)
            # Joined form is a real word → OCR split
            return joined

        text = re.sub(r'\b([A-Za-z]{1,5})\.([a-z]{2,})\b', fix_period_split, text)

        # Fix OCR line-break hyphenation artifacts:
        # "charac- teristic" → "characteristic" (OCR break mid-word)
        # "Zionist- racialists" → "Zionist-racialists" (compound, keep hyphen)
        # Known compound words that should keep their hyphens
        _compound_words = {
            'freedom-loving', 'war-mongering', 'man-hating', 'money-bags',
            'so-called', 'new-styled', 'chauvinist-racist', 'well-known',
            'well-being', 'hard-line', 'hard-won', 'long-standing',
            'peace-loving', 'blood-thirsty', 'cold-blooded', 'ill-fated',
        }
        _compound_prefixes = ('anti', 'non', 'pre', 'post', 'semi', 'self',
                              'co', 'ex', 'neo', 'pro', 'nazi', 'well',
                              'counter', 'cross', 'inter', 'multi', 'over',
                              'super', 'trans', 'under', 'vice')

        def fix_hyphen_break(m):
            word_before = m.group(1)
            after_part = m.group(2)
            full = (word_before + '-' + after_part).lower()

            # Keep hyphen for known compound words
            if full in _compound_words:
                return word_before + '-' + after_part
            # Keep hyphen for known compound prefixes
            if word_before.lower() in _compound_prefixes:
                return word_before + '-' + after_part
            # Keep hyphen if both sides start with uppercase (proper noun compound: Ben-Gurion)
            if after_part[0].isupper() and word_before[0].isupper():
                return word_before + '-' + after_part
            # Keep hyphen if word_before ends in a common word ending (complete word)
            if re.search(r'(?:ist|ism|ial|ing|tion|ment|ness|ous|ive|ant|ent|tic|ler|dom|ful|less|ward|wise|like|fold)$',
                         word_before.lower()):
                return word_before + '-' + after_part
            # Otherwise, join (it's an OCR line-break artifact)
            return word_before + after_part
        text = re.sub(r'\b(\w+)-\s*(\w+)\b', fix_hyphen_break, text)
        # Remove embedded page numbers (e.g., "con100 sequently" from OCR)
        text = re.sub(r'(\w)\d{1,3}\s+([a-z])', r'\1 \2', text)
        # Remove stray *** markers
        text = re.sub(r'\*{2,}', '', text)
        # Collapse multiple spaces
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting. For production, use spaCy."""
        # Split on period, exclamation, question mark followed by space + uppercase
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def is_propaganda_chunk(self, text: str) -> bool:
        """Return True only if the chunk is likely a direct Soviet propaganda CLAIM.

        Filters out quoted material, citation-heavy passages, pure historical
        narrative, and passages dominated by proper nouns / statistics.
        """
        words = text.split()
        total_words = len(words)
        if total_words == 0:
            return False

        # -----------------------------------------------------------------
        # 1. Reject chunks that are mostly quoted material (>50%)
        # -----------------------------------------------------------------
        # Count characters inside any kind of quotation marks
        quote_pairs = [
            ('\u00ab', '\u00bb'),   # « »
            ('\u201c', '\u201d'),   # " "
            ('\u2018', '\u2019'),   # ' '
        ]
        quoted_chars = 0
        for open_q, close_q in quote_pairs:
            inside = False
            for ch in text:
                if ch == open_q:
                    inside = True
                elif ch == close_q:
                    inside = False
                elif inside:
                    quoted_chars += 1
        # Also handle straight quotes: count text between pairs of "
        straight_parts = text.split('"')
        for i in range(1, len(straight_parts), 2):
            quoted_chars += len(straight_parts[i])

        total_chars = len(text)
        if total_chars > 0 and quoted_chars / total_chars > 0.50:
            return False

        # -----------------------------------------------------------------
        # 2. Reject chunks with heavy citation markers (3+ references)
        # -----------------------------------------------------------------
        ref_brackets = re.findall(r'\[\d+\]', text)
        footnote_markers = re.findall(r'(?<!\[)\b\d{1,3}\)', text)
        if len(ref_brackets) + len(footnote_markers) >= 3:
            return False

        # -----------------------------------------------------------------
        # 3. Reject pure historical narrative (3+ dates, no propaganda keywords)
        # -----------------------------------------------------------------
        date_patterns = re.findall(
            r'\b(?:1[0-9]{3}|20[0-2][0-9])\b'           # years like 1897, 1970, 2020
            r'|'
            r'\b\d{1,2}(?:st|nd|rd|th)\s+century\b',     # "7th century"
            text, re.IGNORECASE,
        )
        if len(date_patterns) >= 3:
            # Check for ANY propaganda keyword from the taxonomy
            text_lower = text.lower()
            has_propaganda_keyword = False
            for trope_info in TROPE_TAXONOMY.values():
                for keyword in trope_info["keywords"]:
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    if re.search(pattern, text_lower):
                        has_propaganda_keyword = True
                        break
                if has_propaganda_keyword:
                    break
            if not has_propaganda_keyword:
                return False

        # -----------------------------------------------------------------
        # 4. Reject chunks mostly proper nouns and statistics (>40%)
        # -----------------------------------------------------------------
        capitalized_or_numeric = sum(
            1 for w in words
            if (w[0].isupper() and len(w) > 1) or w.replace(',', '').replace('.', '').replace('%', '').isdigit()
        )
        # Exclude the first word of each sentence from the capitalized count
        # (it's capitalized by grammar, not because it's a proper noun)
        sentences_in_chunk = re.split(r'(?<=[.!?])\s+', text)
        first_words_capital = sum(
            1 for s in sentences_in_chunk
            if s and s.split()[0][0].isupper() and len(s.split()[0]) > 1
        )
        adjusted = max(0, capitalized_or_numeric - first_words_capital)
        if total_words > 0 and adjusted / total_words > 0.40:
            return False

        # -----------------------------------------------------------------
        # 5. Keep chunks with propagandistic language (Soviet editorial voice)
        # -----------------------------------------------------------------
        propaganda_markers = [
            r'\bmust\b', r'\bclearly\b', r'\bobviously\b',
            r'\bit is known that\b', r'\bso-called\b', r'\ballegedly\b',
            r'\bso called\b', r'\bin reality\b', r'\bin fact\b',
            r'\bundeniable\b', r'\bundoubtedly\b', r'\bunquestionable\b',
            r'\bof course\b', r'\bneedless to say\b',
            r'\breactionary\b', r'\bimperialist\b', r'\bchauvin',
            r'\baggress', r'\bexploit', r'\boppres',
            r'\bcolonial', r'\bracis', r'\bfascis',
            r'\bmilitarist', r'\bcharlatans?\b', r'\bperfidy\b',
            r'\bdemagog', r'\bsinister\b', r'\bmachinations?\b',
        ]
        text_lower = text.lower()
        for marker in propaganda_markers:
            if re.search(marker, text_lower):
                return True

        # Check taxonomy keywords as a final keep signal
        for trope_info in TROPE_TAXONOMY.values():
            for keyword in trope_info["keywords"]:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    return True

        # Default: reject (no propaganda signal detected)
        return False

    def process_corpus_file(self, filepath: str, source_meta: dict) -> list[Passage]:
        """Process a single corpus file into passages.

        After chunking and trope classification, applies propaganda filtering
        to remove noise (historical narrative, quoted material, statistics).
        A chunk is kept if is_propaganda_chunk() returns True OR the chunk
        has 2+ trope labels (strong propaganda signal overrides heuristics).
        """
        with open(filepath, "r") as f:
            text = f.read()

        passages = self.chunk_text(text, source_meta)

        # Classify tropes
        for passage in passages:
            passage.trope_labels = self.classify_tropes_keyword(passage)

        # Apply propaganda pre-filter
        total_before = len(passages)
        filtered = [
            p for p in passages
            if self.is_propaganda_chunk(p.text) or len(p.trope_labels) >= 2
        ]

        if total_before > 0 and len(filtered) < total_before:
            kept_pct = len(filtered) / total_before * 100
            print(f"    Propaganda filter: {total_before} -> {len(filtered)} passages ({kept_pct:.0f}% kept)")

        return filtered


# ============================================================
# EMBEDDING ENGINE
# ============================================================

class EmbeddingEngine:
    """Generate and manage embeddings for the corpus."""

    def __init__(self, model_name: str = CONFIG["default_model"]):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Lazy-load the embedding model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_passages(self, passages: list[Passage], batch_size: int = 32) -> list[Passage]:
        """Generate embeddings for a list of passages."""
        self.load_model()

        texts = []
        for p in passages:
            # For E5 models, prefix with "passage:" for indexing
            if "e5" in self.model_name.lower():
                texts.append(f"passage: {p.text}")
            else:
                texts.append(p.text)

        print(f"Embedding {len(texts)} passages...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        for i, passage in enumerate(passages):
            passage.embedding = embeddings[i]

        return passages

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        self.load_model()

        # For E5 models, prefix with "query:" for search
        if "e5" in self.model_name.lower():
            text = f"query: {text}"

        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0]


# ============================================================
# SIMILARITY SEARCH
# ============================================================

class SimilarityEngine:
    """Find matches between Soviet and modern passages."""

    def __init__(self, config=CONFIG):
        self.config = config
        self.soviet_passages: list[Passage] = []
        self.modern_passages: list[Passage] = []

    def index_passages(self, passages: list[Passage]):
        """Add passages to the appropriate index."""
        for p in passages:
            if p.corpus == "soviet":
                self.soviet_passages.append(p)
            else:
                self.modern_passages.append(p)

    def find_soviet_matches(self, query_embedding: np.ndarray,
                            query_tropes: list[str] = None,
                            top_k: int = None) -> list[tuple[Passage, float]]:
        """Find the most similar Soviet passages to a query."""
        if top_k is None:
            top_k = self.config["top_k"]

        if not self.soviet_passages:
            return []

        # Build matrix of Soviet embeddings
        soviet_embeddings = np.array([p.embedding for p in self.soviet_passages])

        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(soviet_embeddings, query_embedding)

        # Trope overlap bonus
        if query_tropes:
            for i, passage in enumerate(self.soviet_passages):
                overlap = set(passage.trope_labels) & set(query_tropes)
                if overlap:
                    # Boost by 5% per overlapping trope
                    similarities[i] = min(1.0, similarities[i] + 0.05 * len(overlap))

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] >= self.config["low_confidence"]:
                results.append((self.soviet_passages[idx], float(similarities[idx])))

        return results

    def classify_confidence(self, score: float, trope_overlap: bool) -> str:
        """Determine confidence tier for a match."""
        if score >= self.config["high_confidence"] and trope_overlap:
            return "high"
        elif score >= self.config["medium_confidence"]:
            return "medium"
        elif score >= self.config["low_confidence"]:
            return "low"
        return "below_threshold"

    def analyze_text(self, text: str, embedding_engine: EmbeddingEngine,
                     processor: CorpusProcessor) -> list[Match]:
        """Full analysis: embed query, find matches, classify."""
        # Create a temporary passage for trope classification
        temp_passage = Passage(
            id="query", text=text, source="query",
            source_title="User Query", author="",
            year=datetime.now().year, language="en", corpus="modern"
        )
        query_tropes = processor.classify_tropes_keyword(temp_passage)

        # Embed the query
        query_embedding = embedding_engine.embed_query(text)

        # Find matches
        soviet_matches = self.find_soviet_matches(
            query_embedding, query_tropes
        )

        # Build Match objects
        matches = []
        for soviet_passage, score in soviet_matches:
            trope_overlap = list(
                set(soviet_passage.trope_labels) & set(query_tropes)
            )
            confidence = self.classify_confidence(score, bool(trope_overlap))

            matches.append(Match(
                soviet_passage=soviet_passage,
                modern_passage=temp_passage,
                similarity_score=score,
                trope_overlap=trope_overlap,
                confidence_tier=confidence,
            ))

        return matches


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(config=CONFIG):
    """Evaluate embedding models on known Soviet-modern pairs.
    
    These are hand-curated pairs where we KNOW the modern text
    echoes the Soviet original. A good model should score these
    higher than random pairs.
    """

    # Known echo pairs (Soviet → Modern)
    KNOWN_PAIRS = [
        # ZIONISM = RACISM
        (
            "The main posits of modern Zionism are militant chauvinism, racism, anti-Communism and anti-Sovietism.",
            "Zionism is a racist, settler-colonial ideology that promotes Jewish supremacy.",
        ),
        # ZIONISM = IMPERIALISM
        (
            "Serving as the front squad of colonialism and neo-colonialism, international Zionism actively participates in the fight against national liberation movements.",
            "Israel is a settler-colonial state built on the dispossession of indigenous Palestinians, and Zionism is a tool of Western imperialism.",
        ),
        # WEAPONIZED ANTISEMITISM
        (
            "Absurd are attempts of Zionist ideologists to present those who criticize them, or condemn the aggressive politics of Israel's ruling circles, as antisemitic.",
            "Accusations of antisemitism are weaponized to silence legitimate criticism of Israel and deflect from its crimes against Palestinians.",
        ),
        # JEWISH CONSPIRACY / MEDIA CONTROL
        (
            "International Zionist Organization influences or controls significant part of media agencies and outlets in the West.",
            "The Zionist lobby controls the mainstream media narrative and uses its influence to suppress Palestinian voices.",
        ),
        # ZIONISM = NAZISM
        (
            "The Hitlerites acted in the same way when they exterminated the inferior Jewish race. Zionism-Fascism-Hitlerism.",
            "What Israel is doing in Gaza is genocide. They are the new Nazis carrying out a holocaust against Palestinians.",
        ),
        # ANTI-ZIONISM AS PROGRESSIVE DUTY
        (
            "We call on all Soviet citizens: workers, peasants, representatives of intelligentsia: take active part in exposing Zionism.",
            "As progressives, we have a moral obligation to stand against Zionism and support the Palestinian liberation struggle through BDS.",
        ),
    ]

    # Negative pairs (should score LOW — legitimate criticism, not propaganda-derived)
    NEGATIVE_PAIRS = [
        (
            "The main posits of modern Zionism are militant chauvinism, racism.",
            "I disagree with the Israeli government's settlement expansion policy in the West Bank.",
        ),
        (
            "Zionism concentrates ultra-nationalism and racial intolerance.",
            "The two-state solution requires compromises from both Israelis and Palestinians.",
        ),
        (
            "International Zionist Organization controls media agencies.",
            "I think US foreign aid to Israel should be reviewed as part of overall budget priorities.",
        ),
    ]

    from sentence_transformers import SentenceTransformer

    results = {}

    for model_name in config["models"]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Compute similarities for known pairs
        known_scores = []
        for soviet_text, modern_text in KNOWN_PAIRS:
            if "e5" in model_name.lower():
                emb_s = model.encode(f"passage: {soviet_text}", normalize_embeddings=True)
                emb_m = model.encode(f"query: {modern_text}", normalize_embeddings=True)
            else:
                emb_s = model.encode(soviet_text, normalize_embeddings=True)
                emb_m = model.encode(modern_text, normalize_embeddings=True)

            score = float(np.dot(emb_s, emb_m))
            known_scores.append(score)

        # Compute similarities for negative pairs
        negative_scores = []
        for soviet_text, modern_text in NEGATIVE_PAIRS:
            if "e5" in model_name.lower():
                emb_s = model.encode(f"passage: {soviet_text}", normalize_embeddings=True)
                emb_m = model.encode(f"query: {modern_text}", normalize_embeddings=True)
            else:
                emb_s = model.encode(soviet_text, normalize_embeddings=True)
                emb_m = model.encode(modern_text, normalize_embeddings=True)

            score = float(np.dot(emb_s, emb_m))
            negative_scores.append(score)

        avg_known = np.mean(known_scores)
        avg_negative = np.mean(negative_scores)
        separation = avg_known - avg_negative

        results[model_name] = {
            "avg_known_pair_score": avg_known,
            "avg_negative_pair_score": avg_negative,
            "separation": separation,
            "known_scores": known_scores,
            "negative_scores": negative_scores,
            "dim": model.get_sentence_embedding_dimension(),
        }

        print(f"  Known pairs (should be HIGH):    avg={avg_known:.4f}  scores={[f'{s:.3f}' for s in known_scores]}")
        print(f"  Negative pairs (should be LOW):   avg={avg_negative:.4f}  scores={[f'{s:.3f}' for s in negative_scores]}")
        print(f"  Separation (higher is better):    {separation:.4f}")
        print(f"  Embedding dimension:              {results[model_name]['dim']}")

        del model  # Free memory

    # Rank models
    print(f"\n{'='*60}")
    print("MODEL RANKING (by separation between known and negative pairs)")
    print(f"{'='*60}")
    ranked = sorted(results.items(), key=lambda x: x[1]["separation"], reverse=True)
    for i, (name, r) in enumerate(ranked):
        print(f"  {i+1}. {name}")
        print(f"     Separation: {r['separation']:.4f} | Known avg: {r['avg_known_pair_score']:.4f} | Negative avg: {r['avg_negative_pair_score']:.4f}")

    return results


# ============================================================
# DEMO / QUICK TEST
# ============================================================

def run_demo():
    """Quick demo using sample data — run this first to verify everything works."""

    print("="*60)
    print("I'VE SEEN THIS BEFORE — Demo")
    print("="*60)

    # Sample Soviet passages
    soviet_texts = [
        {
            "text": "Modern Zionism is the ideology, a ramified system of organisations and the practical politics of the wealthy Jewish bourgeoisie which has closely allied itself with monopoly circles in the USA and other imperialist countries. The main content of Zionism is bellicose chauvinism and anti-communism.",
            "source": "ivanov_1970", "title": "Caution: Zionism!", "author": "Yuri Ivanov", "year": 1970,
        },
        {
            "text": "The main posits of modern Zionism are militant chauvinism, racism, anti-Communism and anti-Sovietism. The anti-human reactionary essence of Zionism is overt and covert fight against freedom movements and against the USSR.",
            "source": "great_soviet_encyclopedia", "title": "Great Soviet Encyclopedia", "author": "CPSU", "year": 1975,
        },
        {
            "text": "International Zionist Organization owns major financial funds, partly through Jewish monopolists and partly collected by Jewish mandatory charities. It also influences or controls significant part of media agencies and outlets in the West.",
            "source": "great_soviet_encyclopedia", "title": "Great Soviet Encyclopedia", "author": "CPSU", "year": 1975,
        },
        {
            "text": "Serving as the front squad of colonialism and neo-colonialism, international Zionism actively participates in the fight against national liberation movements of the peoples of Africa, Asia and Latin America.",
            "source": "great_soviet_encyclopedia", "title": "Great Soviet Encyclopedia", "author": "CPSU", "year": 1975,
        },
        {
            "text": "By its nature, Zionism concentrates ultra-nationalism, chauvinism and racial intolerance, excuse for territorial occupation and annexation, military opportunism, cult of political promiscuousness and irresponsibility, demagogy and ideological diversion, dirty tactics and perfidy.",
            "source": "anti_zionist_committee_1983", "title": "Anti-Zionist Committee Declaration", "author": "CPSU/KGB", "year": 1983,
        },
        {
            "text": "Absurd are attempts of Zionist ideologists to present those who criticize them, or condemn the aggressive politics of Israel's ruling circles, as antisemitic.",
            "source": "anti_zionist_committee_1983", "title": "Anti-Zionist Committee Declaration", "author": "CPSU/KGB", "year": 1983,
        },
        {
            "text": "We call on all Soviet citizens: workers, peasants, representatives of intelligentsia: take active part in exposing Zionism, strongly rebuke its endeavors.",
            "source": "anti_zionist_committee_1983", "title": "Anti-Zionist Committee Declaration", "author": "CPSU/KGB", "year": 1983,
        },
        {
            "text": "The Zionist Concern is at the same time one of the world's largest associations of finance capital, a self-styled global ministry for the affairs of world Jewry, an international intelligence centre, and a well-run misinformation and propaganda service.",
            "source": "ivanov_1970", "title": "Caution: Zionism!", "author": "Yuri Ivanov", "year": 1970,
        },
        {
            "text": "The Hitlerites acted in the same way when they exterminated the inferior Jewish race. His ugly statements about man-hating Zionism and his equation, Zionism-Fascism-Hitlerism.",
            "source": "novick_1983", "title": "The Anti-Zionist Campaign", "author": "Paul Novick (documenting Soviet propaganda)", "year": 1983,
        },
        {
            "text": "Having kidnapped the right of defenders of Soviet Jews, those Zionist wheeler-dealers try to persuade the world's public opinion that, allegedly, in the USSR there exists the Jewish question.",
            "source": "anti_zionist_committee_1983", "title": "Anti-Zionist Committee Declaration", "author": "CPSU/KGB", "year": 1983,
        },
    ]

    # Sample modern texts to test against
    modern_test_texts = [
        "Zionism is a racist, settler-colonial ideology. Israel is an apartheid state that practices ethnic cleansing against Palestinians.",
        "The Zionist lobby controls American foreign policy and silences any criticism of Israel by smearing critics as antisemitic.",
        "From the river to the sea, Palestine will be free. Zionism is a form of white supremacy and must be dismantled.",
        "Israel is doing to Palestinians what the Nazis did to Jews. Gaza is an open-air concentration camp.",
        "As progressives, we must boycott, divest, and sanction Israel. Anti-Zionism is not antisemitism.",
        "I support a two-state solution and think both sides need to make compromises for peace.",  # Legitimate criticism — should score LOW
        "The Israeli government's settlement expansion undermines the peace process.",  # Legitimate criticism — should score LOW
    ]

    # Initialize components
    processor = CorpusProcessor()
    engine = EmbeddingEngine()
    similarity = SimilarityEngine()

    # Create Soviet passages
    print("\n--- Creating Soviet passages ---")
    soviet_passages = []
    for item in soviet_texts:
        p = Passage(
            id=f"{item['source']}_{len(soviet_passages):04d}",
            text=item["text"],
            source=item["source"],
            source_title=item["title"],
            author=item["author"],
            year=item["year"],
            language="en",
            corpus="soviet",
        )
        p.trope_labels = processor.classify_tropes_keyword(p)
        soviet_passages.append(p)
        print(f"  [{p.id}] Tropes: {p.trope_labels}")

    # Embed Soviet passages
    print("\n--- Embedding Soviet passages ---")
    soviet_passages = engine.embed_passages(soviet_passages)
    similarity.index_passages(soviet_passages)

    # Test each modern text
    print("\n" + "="*60)
    print("MATCH RESULTS")
    print("="*60)

    for modern_text in modern_test_texts:
        print(f"\n{'─'*60}")
        print(f"MODERN TEXT: \"{modern_text[:100]}...\"")
        print(f"{'─'*60}")

        matches = similarity.analyze_text(modern_text, engine, processor)

        if not matches:
            print("  No matches above threshold.")
            continue

        for i, match in enumerate(matches[:3]):  # Show top 3
            print(f"\n  Match {i+1} [{match.confidence_tier.upper()}] (score: {match.similarity_score:.4f})")
            print(f"  Soviet source: {match.soviet_passage.source_title} ({match.soviet_passage.year})")
            print(f"  Soviet text: \"{match.soviet_passage.text[:150]}...\"")
            if match.trope_overlap:
                print(f"  Shared tropes: {match.trope_overlap}")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline():
    """Run the complete corpus processing and analysis pipeline."""

    config = CONFIG
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["index_dir"], exist_ok=True)

    # --- Step 1: Process Soviet corpus ---
    print("\n=== Step 1: Processing Soviet corpus ===")
    processor = CorpusProcessor(config)

    soviet_sources = {
        "anti_zionist_committee_declaration_1983.txt": {
            "source": "anti_zionist_committee_1983",
            "title": "Anti-Zionist Committee of the Soviet Public Declaration",
            "author": "CPSU / KGB",
            "year": 1983,
            "language": "en",
            "corpus": "soviet",
        },
        "great_soviet_encyclopedia_zionism.txt": {
            "source": "great_soviet_encyclopedia",
            "title": "Great Soviet Encyclopedia - Zionism Entry",
            "author": "CPSU",
            "year": 1975,
            "language": "en",
            "corpus": "soviet",
        },
        # "ivanov_caution_zionism_1970_excerpts.txt": {
        "ivanov_full.txt": {
            "source": "ivanov_1970",
            "title": "Caution: Zionism!",
            "author": "Yuri Ivanov",
            "year": 1970,
            "language": "en",
            "corpus": "soviet",
        },
        "novick_anti_zionist_campaign_1983.txt": {
            "source": "novick_1983",
            "title": "The Anti-Zionist Campaign in the USSR",
            "author": "Paul Novick (documenting Soviet propaganda)",
            "year": 1983,
            "language": "en",
            "corpus": "soviet",
        },
        "novosti_instrument_imperialist_reaction_1970.txt": {
            "source": "novosti_1970",
            "title": "Zionism: Instrument of Imperialist Reaction",
            "author": "Novosti Press Agency",
            "year": 1970,
            "language": "en",
            "corpus": "soviet",
        },
        "anti_zionist_committee_aims_tasks_1983.txt": {
            "source": "azc_1983",
            "title": "Anti-Zionist Committee: Aims and Tasks",
            "author": "CPSU / KGB",
            "year": 1983,
            "language": "en",
            "corpus": "soviet",
        },
        "zionism_enemy_peace_progress_1985.txt": {
            "source": "progress_1985",
            "title": "Zionism: Enemy of Peace and Social Progress",
            "author": "Progress Publishers",
            "year": 1985,
            "language": "en",
            "corpus": "soviet",
        },
    }

    all_passages = []
    corpus_dir = Path(config["corpus_dir"])

    for filename, meta in soviet_sources.items():
        filepath = corpus_dir / filename
        if filepath.exists():
            passages = processor.process_corpus_file(str(filepath), meta)
            all_passages.extend(passages)
            print(f"  Processed {filename}: {len(passages)} passages")
        else:
            print(f"  WARNING: {filename} not found in {corpus_dir}")

    print(f"\nTotal Soviet passages: {len(all_passages)}")

    # --- Step 2: Embed passages ---
    print("\n=== Step 2: Generating embeddings ===")
    engine = EmbeddingEngine(config["default_model"])
    all_passages = engine.embed_passages(all_passages)

    # --- Step 3: Save index ---
    print("\n=== Step 3: Saving index ===")
    index_path = Path(config["index_dir"]) / "soviet_index.json"
    index_data = [p.to_dict() for p in all_passages]
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2, default=str)
    print(f"  Saved {len(all_passages)} passages to {index_path}")

    # --- Step 4: Save embeddings as numpy ---
    embeddings_path = Path(config["index_dir"]) / "soviet_embeddings.npy"
    embeddings_matrix = np.array([p.embedding for p in all_passages])
    np.save(embeddings_path, embeddings_matrix)
    print(f"  Saved embeddings matrix: {embeddings_matrix.shape} to {embeddings_path}")

    # --- Step 5: Trope distribution summary ---
    print("\n=== Trope Distribution ===")
    trope_counts = {}
    for p in all_passages:
        for t in p.trope_labels:
            trope_counts[t] = trope_counts.get(t, 0) + 1

    for trope_id, count in sorted(trope_counts.items(), key=lambda x: -x[1]):
        name = TROPE_TAXONOMY.get(trope_id, {}).get("name", trope_id)
        print(f"  {name}: {count} passages")

    print("\n=== Pipeline complete ===")
    print(f"Next steps:")
    print(f"  1. Download full Ivanov text and re-run")
    print(f"  2. Run model evaluation: python {sys.argv[0]} --eval-only")
    print(f"  3. Test with modern texts: python {sys.argv[0]} --demo")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if "--eval-only" in sys.argv:
        evaluate_models()
    elif "--demo" in sys.argv:
        run_demo()
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        run_full_pipeline()
