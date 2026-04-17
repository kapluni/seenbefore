"""
process_modern_sources.py — ETL for downloaded modern datasets
==============================================================
Reads ISCA, CONAN, and other datasets, filters for antisemitic/anti-Zionist
content, and outputs corpus/modern_corpus.json for the embedding pipeline.

Usage:  python process_modern_sources.py
Input:  corpus/modern_sources/  (populated by download_datasets.sh)
Output: corpus/modern_corpus.json
"""

import os, csv, json, re
from pathlib import Path
from collections import Counter

def normalize_for_dedup(text):
    """Normalize text for near-duplicate detection: lowercase, strip articles/mentions/urls, collapse whitespace."""
    t = text.lower()
    t = re.sub(r'https?://\S+', '', t)           # Remove URLs
    t = re.sub(r'@\w+', '', t)                    # Remove @mentions
    t = re.sub(r'#(\w+)', r'\1', t)               # Remove # but keep word
    t = re.sub(r'\b(the|a|an|is|are|was|were)\b', '', t)  # Strip common articles/copulas
    t = re.sub(r'[^\w\s]', ' ', t)                # Remove punctuation
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def clean_tweet_text(text):
    """Clean tweet text for better embedding: remove @mentions, URLs, fix encoding."""
    t = text
    # Fix literal \n (backslash + n, 0x5c 0x6e) from JSON-escaped tweets
    t = t.replace('\\n', ' ').replace('\n', ' ')
    # Fix HTML entities
    t = t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Fix UTF-8 mojibake (common in tweets scraped with wrong encoding)
    t = t.replace('\u00e2\u0080\u009c', '"').replace('\u00e2\u0080\u009d', '"')
    t = t.replace('\u00e2\u0080\u0099', "'").replace('\u00e2\u0080\u0098', "'")
    t = t.replace('\u00e2\u0080\u0093', '–').replace('\u00e2\u0080\u0094', '—')
    t = t.replace('\u00e2\u0080\u00a6', '…')
    t = re.sub(r'â€[œ"]', '"', t)
    t = re.sub(r'â€[™˜]', "'", t)
    t = re.sub(r'â€¦', '…', t)
    t = re.sub(r'â€["\u0093\u0094]', '–', t)
    t = re.sub(r'â€\S?', '', t)
    # Remove emoji mojibake (e.g., ðŸ'ðŸ' sequences — UTF-8 emoji decoded as latin-1)
    t = re.sub(r'\u00f0[\u0080-\u024f]{1,3}', '', t)
    t = re.sub(r'[\xc3\xc2][\x80-\xbf]', '', t)  # broken 2-byte UTF-8 sequences
    t = re.sub(r'Ã[^\s]*', '', t)  # Ã-prefixed mojibake
    # Remove orphan high bytes left after mojibake cleanup (control chars, private use)
    t = re.sub(r'[\x80-\x9f]', '', t)
    # Remove @mentions, URLs, RT prefix
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'RT\s+', '', t)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_propaganda_from_incident(text):
    """Extract quoted propaganda content from HEATMap incident descriptions.
    E.g., 'posted stickers that read: \"No Zionists in government\"' → 'No Zionists in government'
    Falls back to full text if no quotes found."""
    # Try to find quoted content after read/said/stated/posted
    patterns = [
        r'(?:read|said|stated|wrote|posted|displayed|reading)[\s:]*["\u201c]([^"\u201d]+)["\u201d]',
        r'(?:read|said|stated|wrote|posted|displayed|reading)[\s:]*\'([^\']+)\'',
        r'"([^"]{20,})"',  # Any substantial quoted text
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m and len(m.group(1).strip()) >= 15:
            return m.group(1).strip()
    return text

OUTPUT_PATH = "corpus/modern_corpus.json"

def process_isca_zenodo(filepath):
    passages = []
    if not os.path.exists(filepath):
        print(f"  ⚠ Not found: {filepath}"); return passages
    print(f"  Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        print(f"  Columns: {fields}")
        text_col = next((c for c in fields if c.lower().strip() in ("text","tweet","tweet_text","content","message")), None)
        bias_col = next((c for c in fields if c.lower().strip() in ("biased","antisemitic","label","bias","is_antisemitic")), None)
        keyword_col = next((c for c in fields if c.lower().strip() in ("keyword","keywords","query")), None)
        if not text_col:
            print(f"  ⚠ No text column found"); return passages
        total = kept = 0
        for row in reader:
            total += 1
            text = clean_tweet_text((row.get(text_col) or "").strip())
            if not text or len(text) < 20: continue
            if bias_col:
                v = (row.get(bias_col) or "").strip().lower()
                if v not in ("1","1.0","yes","true","antisemitic","biased"): continue
            passages.append({"text": text, "source": "isca_zenodo", "source_title": "ISCA Twitter Dataset (Zenodo)", "year": 2021, "language": "en", "is_antisemitic": True, "keyword": (row.get(keyword_col) or "").strip() if keyword_col else ""})
            kept += 1
        print(f"  {total} rows → {kept} antisemitic tweets")
    return passages

def process_isca_huggingface(dirpath):
    passages = []
    for fname in ["Antisemitism_dataset.csv", "DatasetForMachineLearning.csv"]:
        fp = os.path.join(dirpath, fname)
        if not os.path.exists(fp): continue
        print(f"  Reading {fp}...")
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            # Match column names exactly (avoid "TweetID" matching "text")
            text_col = next((c for c in fields if c.lower().strip() in ("text","tweet","tweet_text","content","message")), None)
            bias_col = next((c for c in fields if c.lower().strip() in ("biased","bias","antisemitic","label","is_antisemitic")), None)
            keyword_col = next((c for c in fields if c.lower().strip() in ("keyword","keywords","target")), None)
            if not text_col: continue
            seen, total, kept = set(), 0, 0
            for row in reader:
                total += 1
                # Filter for Jewish-targeted content in multi-group datasets
                if keyword_col:
                    kw = (row.get(keyword_col) or "").strip().lower()
                    if kw not in ("jews", "jewish", "antisemitism", "zionism"): continue
                text = clean_tweet_text((row.get(text_col) or "").strip())
                if not text or len(text) < 20: continue
                h = hash(text[:100])
                if h in seen: continue
                seen.add(h)
                if bias_col:
                    v = (row.get(bias_col) or "").strip().lower()
                    if v not in ("1","1.0","yes","true","antisemitic","biased"): continue
                passages.append({"text": text, "source": "isca_huggingface", "source_title": "ISCA HateSpeech & Bias (HuggingFace)", "year": 2023, "language": "en", "is_antisemitic": True})
                kept += 1
            print(f"  {total} rows → {kept} antisemitic")
    return passages

def process_conan(filepath):
    passages = []
    if not os.path.exists(filepath):
        print(f"  ⚠ Not found: {filepath}"); return passages
    print(f"  Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        col = {}
        for c in fields:
            cl = c.lower().strip()
            if "hate" in cl and "speech" in cl: col["hate"] = c
            elif "counter" in cl: col["counter"] = c
            elif "target" in cl: col["target"] = c
        if "hate" not in col:
            print("  ⚠ No hate speech column"); return passages
        seen, total, jews = set(), 0, 0
        for row in reader:
            total += 1
            target = (row.get(col.get("target",""),"") or "").upper()
            if "JEW" not in target: continue
            jews += 1
            text = clean_tweet_text((row.get(col["hate"],"") or "").strip())
            cn = (row.get(col.get("counter",""),"") or "").strip()
            if not text or len(text) < 10: continue
            h = hash(text[:80])
            if h in seen: continue
            seen.add(h)
            passages.append({"text": text, "source": "conan_multitarget", "source_title": "CONAN Counter-Narrative Dataset", "year": 2022, "language": "en", "is_antisemitic": True, "counter_narrative": cn})
        print(f"  {total} rows, {jews} Jewish-targeted, {len(passages)} unique")
    return passages

def process_conan_dialogues(filepath):
    passages = []
    if not os.path.exists(filepath): return passages
    print(f"  Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        target_col = next((c for c in fields if "target" in c.lower()), None)
        text_col = next((c for c in fields if "text" in c.lower()), None)
        type_col = next((c for c in fields if "type" in c.lower()), None)
        if not text_col: return passages
        seen = set()
        for row in reader:
            target = (row.get(target_col,"") or "").upper() if target_col else ""
            mtype = (row.get(type_col,"") or "").upper() if type_col else ""
            if "JEW" not in target: continue
            if mtype and "HS" not in mtype and "HATE" not in mtype: continue
            text = clean_tweet_text((row.get(text_col,"") or "").strip())
            if not text or len(text) < 15: continue
            h = hash(text[:80])
            if h in seen: continue
            seen.add(h)
            passages.append({"text": text, "source": "conan_dialogue", "source_title": "CONAN Dialogue Dataset", "year": 2022, "language": "en", "is_antisemitic": True})
        print(f"  Kept {len(passages)} Jewish-targeted hate speech from dialogues")
    return passages

def process_goldstandard(filepath):
    """Process GoldStandard2024 Twitter bias dataset — filter for Jewish/Israel-targeted biased tweets."""
    passages = []
    if not os.path.exists(filepath):
        print(f"  ⚠ Not found: {filepath}"); return passages
    print(f"  Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        print(f"  Columns: {fields}")
        text_col = next((c for c in fields if c.lower().strip() in ("text","tweet","tweet_text","content","message")), None)
        bias_col = next((c for c in fields if c.lower().strip() in ("biased","bias","antisemitic","label","is_antisemitic")), None)
        keyword_col = next((c for c in fields if c.lower().strip() in ("keyword","keywords","query")), None)
        if not text_col:
            print(f"  ⚠ No text column found"); return passages
        jewish_keywords = {"jews", "jewish", "kikes", "zionazi", "israel"}
        seen, total, kept = set(), 0, 0
        for row in reader:
            total += 1
            # Filter for Jewish/Israel-targeted content
            if keyword_col:
                kw = (row.get(keyword_col) or "").strip().lower()
                if kw not in jewish_keywords: continue
            # Filter for biased tweets only
            if bias_col:
                v = (row.get(bias_col) or "").strip().lower()
                if v not in ("1", "1.0", "yes", "true", "antisemitic", "biased"): continue
            text = clean_tweet_text((row.get(text_col) or "").strip())
            if not text or len(text) < 20: continue
            h = hash(text[:100])
            if h in seen: continue
            seen.add(h)
            passages.append({
                "text": text,
                "source": "goldstandard_2024",
                "source_title": "GoldStandard 2024 Twitter Bias Dataset",
                "year": 2024,
                "language": "en",
                "is_antisemitic": True,
                "keyword": (row.get(keyword_col) or "").strip() if keyword_col else ""
            })
            kept += 1
        print(f"  {total} rows → {kept} biased Jewish/Israel-targeted tweets")
    return passages

def process_isca_classdata(filepath):
    """Process ISCA ClassData2022and2023.csv — filter for Jewish-targeted biased tweets."""
    passages = []
    if not os.path.exists(filepath):
        print(f"  ⚠ Not found: {filepath}"); return passages
    print(f"  Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        print(f"  Columns: {fields}")
        text_col = next((c for c in fields if c.lower().strip() in ("text","tweet","tweet_text","content","message")), None)
        bias_col = next((c for c in fields if c.lower().strip() in ("biased","bias","antisemitic","label","is_antisemitic")), None)
        keyword_col = next((c for c in fields if c.lower().strip() in ("keyword","keywords","query")), None)
        if not text_col:
            print(f"  ⚠ No text column found"); return passages
        jewish_keywords = {"jews", "jewish"}
        seen, total, kept = set(), 0, 0
        for row in reader:
            total += 1
            # Filter for Jewish-targeted content
            if keyword_col:
                kw = (row.get(keyword_col) or "").strip().lower()
                if kw not in jewish_keywords: continue
            # Filter for biased tweets only
            if bias_col:
                v = (row.get(bias_col) or "").strip().lower()
                if v not in ("1", "1.0", "yes", "true", "antisemitic", "biased"): continue
            text = clean_tweet_text((row.get(text_col) or "").strip())
            if not text or len(text) < 20: continue
            h = hash(text[:100])
            if h in seen: continue
            seen.add(h)
            passages.append({
                "text": text,
                "source": "isca_classdata_2022_2023",
                "source_title": "ISCA ClassData 2022-2023 (Zenodo)",
                "year": 2023,
                "language": "en",
                "is_antisemitic": True,
                "keyword": (row.get(keyword_col) or "").strip() if keyword_col else ""
            })
            kept += 1
        print(f"  {total} rows → {kept} biased Jewish-targeted tweets")
    return passages


def process_hf_hate_superset(dirpath):
    """Process English Hate Speech Superset from HuggingFace — filter for Jewish-targeted content.
    Requires HF_TOKEN env var and access approval at:
    https://huggingface.co/datasets/manueltonneau/english-hate-speech-superset
    """
    csv_path = os.path.join(dirpath, "hate_superset_jewish.csv")
    if os.path.exists(csv_path):
        # If we've already downloaded and filtered, just read the CSV
        print(f"  Reading cached {csv_path}...")
        passages = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = clean_tweet_text((row.get("text") or "").strip())
                if not text or len(text) < 20: continue
                passages.append({
                    "text": text,
                    "source": "hf_hate_superset",
                    "source_title": "English Hate Speech Superset (HuggingFace)",
                    "year": 2024,
                    "language": "en",
                    "is_antisemitic": True,
                    "dataset_source": (row.get("dataset") or "").strip(),
                })
        print(f"  {len(passages)} Jewish-targeted passages from cache")
        return passages

    # Try to download from HuggingFace (requires authentication)
    try:
        from datasets import load_dataset
        print("  Loading from HuggingFace (requires HF_TOKEN)...")
        ds = load_dataset("manueltonneau/english-hate-speech-superset", split="train")
        print(f"  Total rows: {len(ds)}")
        # Filter: labels==1 (hateful) and text contains Jewish-related keywords
        jewish_kw = re.compile(r'\b(jew|jews|jewish|zion|zionist|antisemit|kike|hebrew|israel|idf|iof|apartheid.*israel|palestine)\b', re.IGNORECASE)
        passages = []
        kept = 0
        os.makedirs(dirpath, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=["text", "labels", "source", "dataset"])
            writer.writeheader()
            for row in ds:
                if row.get("labels") != 1: continue
                text = clean_tweet_text((row.get("text") or "").strip())
                if not text or len(text) < 20: continue
                if not jewish_kw.search(text): continue
                writer.writerow({"text": text, "labels": row.get("labels"), "source": row.get("source",""), "dataset": row.get("dataset","")})
                passages.append({
                    "text": text,
                    "source": "hf_hate_superset",
                    "source_title": "English Hate Speech Superset (HuggingFace)",
                    "year": 2024,
                    "language": "en",
                    "is_antisemitic": True,
                    "dataset_source": (row.get("dataset") or ""),
                })
                kept += 1
        print(f"  Filtered {kept} Jewish-targeted hateful posts, saved to {csv_path}")
        return passages
    except Exception as e:
        print(f"  ⚠ Could not load HF dataset: {e}")
        print("  → Set HF_TOKEN env var and request access at https://huggingface.co/datasets/manueltonneau/english-hate-speech-superset")
        return []


def process_heat_map(filepath):
    """Process ADL H.E.A.T. Map incident data — filter for Israel/Zionism-related incidents."""
    passages = []
    if not os.path.exists(filepath):
        print(f"  ⚠ Not found: {filepath}"); return passages
    print(f"  Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        # Strip BOM from first field name
        fields = [f.lstrip('\ufeff') for f in fields]
        print(f"  Columns: {fields}")
        # Find columns
        desc_col = next((c for c in reader.fieldnames or [] if "description" in c.lower()), None)
        iz_col = next((c for c in reader.fieldnames or [] if "israel" in c.lower() or "zionism" in c.lower()), None)
        date_col = next((c for c in reader.fieldnames or [] if c.lower().strip() == "date"), None)
        type_col = next((c for c in reader.fieldnames or [] if c.lower().strip() == "type"), None)
        if not desc_col:
            print(f"  ⚠ No description column found"); return passages
        seen, total, kept = set(), 0, 0
        for row in reader:
            total += 1
            # Filter for Israel/Zionism-related incidents
            if iz_col:
                v = (row.get(iz_col) or "").strip()
                if v != "1": continue
            raw_text = clean_tweet_text((row.get(desc_col) or "").strip())
            if not raw_text or len(raw_text) < 20: continue
            # Extract quoted propaganda content from incident descriptions
            text = extract_propaganda_from_incident(raw_text)
            if len(text) < 15: continue
            h = hash(text[:100])
            if h in seen: continue
            seen.add(h)
            # Extract year from date field (format: MM/DD/YYYY)
            year = 2024
            if date_col:
                date_str = (row.get(date_col) or "").strip()
                if "/" in date_str:
                    parts = date_str.split("/")
                    if len(parts) == 3 and len(parts[2]) == 4:
                        try: year = int(parts[2])
                        except ValueError: pass
            incident_type = (row.get(type_col) or "").strip() if type_col else ""
            passages.append({
                "text": text,
                "source": "adl_heat_map",
                "source_title": "ADL H.E.A.T. Map",
                "year": year,
                "language": "en",
                "is_antisemitic": True,
                "incident_type": incident_type
            })
            kept += 1
        print(f"  {total} rows → {kept} Israel/Zionism-related incidents")
    return passages

def main():
    print("="*60 + "\nPROCESSING MODERN SOURCES\n" + "="*60)
    all_p = []
    print("\n[1] ISCA Zenodo (GoldStandard 2024, 11K tweets)"); all_p.extend(process_isca_zenodo("corpus/modern_sources/isca_zenodo/data.csv"))
    print("\n[2] ISCA ClassData 2022-2023"); all_p.extend(process_isca_classdata("corpus/modern_sources/isca_classdata/ClassData2022and2023.csv"))
    print("\n[3] ISCA HuggingFace"); all_p.extend(process_isca_huggingface("corpus/modern_sources/isca_huggingface"))
    print("\n[4] CONAN Multitarget"); all_p.extend(process_conan("corpus/modern_sources/conan/Multitarget-CONAN.csv"))
    print("\n[5] CONAN Dialogues"); all_p.extend(process_conan_dialogues("corpus/modern_sources/conan/DIALOCONAN.csv"))
    print("\n[6] HF English Hate Speech Superset"); all_p.extend(process_hf_hate_superset("corpus/modern_sources/hf_hate_superset"))
    print("\n[7] ADL H.E.A.T. Map"); all_p.extend(process_heat_map("corpus/new_sources/HEATMapData.csv"))
    # Deduplicate using normalized text to catch near-duplicates
    seen, deduped = set(), []
    for p in all_p:
        k = normalize_for_dedup(p["text"][:120])
        if k not in seen: seen.add(k); deduped.append(p)
    for i, p in enumerate(deduped): p["id"] = f"modern_{i:05d}"
    print(f"\n{'='*60}\nMODERN CORPUS: {len(deduped)} passages (deduped from {len(all_p)})")
    for src, cnt in Counter(p["source"] for p in deduped).most_common():
        print(f"  {src}: {cnt}")
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f: json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
