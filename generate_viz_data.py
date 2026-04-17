"""
generate_viz_data.py — Bridge between embedding pipeline and React visualization
=================================================================================
Loads Soviet + modern corpora, embeds both, computes cross-corpus similarity,
outputs viz_data.json for the frontend.

Usage:
    python generate_viz_data.py --generate
    python generate_viz_data.py --generate --max-modern 500
    python generate_viz_data.py --serve
"""
import json, sys, os, argparse, re, concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import ftfy
from embedding_pipeline import CONFIG, TROPE_TAXONOMY, Passage, CorpusProcessor, EmbeddingEngine, SimilarityEngine

CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-large"
CROSS_ENCODER_THRESHOLD = 0.35  # Below this = not a genuine argument match (unless claim similarity is very high)

CLAIMS_CACHE_PATH = "corpus/claims_cache.json"

# SemEval propaganda techniques for technique overlap detection
PROPAGANDA_TECHNIQUES = [
    "loaded language", "name calling or labeling", "appeal to fear or prejudice",
    "flag waving", "whataboutism", "black and white fallacy",
    "appeal to authority", "bandwagon or appeal to popularity",
    "causal oversimplification", "casting doubt", "exaggeration or minimization",
    "repetition", "slogans", "thought terminating cliches", "straw man", "red herring",
]

def load_claims_cache():
    """Load cached claim extractions from disk."""
    if os.path.exists(CLAIMS_CACHE_PATH):
        with open(CLAIMS_CACHE_PATH) as f:
            return json.load(f)
    return {}

def save_claims_cache(cache):
    """Save claim extractions to disk."""
    os.makedirs(os.path.dirname(CLAIMS_CACHE_PATH) or ".", exist_ok=True)
    with open(CLAIMS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def extract_claims_llm(text, source_type="propaganda", client=None):
    """Use Claude to decompose a passage into atomic claims. Returns list of strings."""
    if not client:
        return []
    context = {"propaganda": "Soviet anti-Zionist propaganda text",
               "echo": "modern anti-Zionist text",
               "criticism": "political commentary"}.get(source_type, "text")
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""Extract the distinct atomic claims from this {context}. Each claim should be:
- A single, self-contained assertion
- Expressible in one sentence
- Specific enough to be matched against other texts
- ACTUALLY STATED in the text below — do NOT infer, extrapolate, or add claims from background knowledge

IMPORTANT: Only extract claims that are explicitly made in the text. Do not add context, historical events, or claims from other sources. If the text says "Zionism is imperialism", extract that — do not add claims about specific events unless the text mentions them.

Return ONLY a JSON array of strings, no other text. If the text contains no clear claims, return [].

Text: "{text}"
"""}])
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        claims = json.loads(content)
        return claims if isinstance(claims, list) else []
    except Exception as e:
        print(f"    WARNING: Claim extraction failed: {e}")
        return []

def extract_claims_batch(texts, source_type, client, cache, cache_prefix=""):
    """Extract claims for a batch of texts, using cache where available."""
    import time
    all_claims = {}
    extracted = 0
    for i, text in enumerate(texts):
        cache_key = f"{cache_prefix}:{text[:200]}"
        if cache_key in cache:
            all_claims[i] = cache[cache_key]
        else:
            claims = extract_claims_llm(text, source_type, client)
            all_claims[i] = claims
            cache[cache_key] = claims
            extracted += 1
            if extracted % 10 == 0:
                print(f"    Extracted claims for {extracted} texts...")
            time.sleep(0.3)  # Rate limiting
    if extracted > 0:
        print(f"    Extracted claims for {extracted} new texts ({len(texts) - extracted} from cache)")
    return all_claims

def compute_claim_similarity(engine, soviet_claims, modern_claims):
    """Compute max claim-pair similarity between two sets of claims.
    Returns (max_score, best_soviet_claim, best_modern_claim)."""
    if not soviet_claims or not modern_claims:
        return 0.0, "", ""
    soviet_embs = engine.model.encode(soviet_claims, normalize_embeddings=True)
    modern_embs = engine.model.encode(modern_claims, normalize_embeddings=True)
    sim_matrix = np.dot(modern_embs, soviet_embs.T)
    best_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    best_score = float(sim_matrix[best_idx])
    return best_score, soviet_claims[best_idx[1]], modern_claims[best_idx[0]]

def detect_techniques(classifier, text, threshold=0.5):
    """Detect propaganda techniques in a text using zero-shot classification."""
    result = classifier(text, PROPAGANDA_TECHNIQUES, multi_label=True)
    return {l: round(float(s), 3) for l, s in zip(result["labels"], result["scores"]) if s >= threshold}

def compute_technique_overlap(techs_a, techs_b):
    """Compute Jaccard overlap between two technique sets."""
    set_a, set_b = set(techs_a.keys()), set(techs_b.keys())
    if not set_a and not set_b:
        return 0.0, []
    shared = sorted(set_a & set_b)
    return len(set_a & set_b) / len(set_a | set_b), shared

def load_cross_encoder():
    """Load the cross-encoder for reranking. Returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        print(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        ce = CrossEncoder(CROSS_ENCODER_MODEL)
        return ce
    except Exception as e:
        print(f"  WARNING: Could not load cross-encoder: {e}")
        return None

def rerank_matches(matches, cross_encoder, threshold=CROSS_ENCODER_THRESHOLD):
    """Rerank matches using cross-encoder to filter for genuine argument matches.
    Returns only matches where the cross-encoder score >= threshold."""
    if not cross_encoder or not matches:
        return matches

    pairs = [(m["sovietTextFull"] or m["sovietText"], m["modernTextFull"] or m["modernText"]) for m in matches]
    print(f"  Cross-encoder scoring {len(pairs)} candidates...")
    scores = cross_encoder.predict(pairs)

    kept = []
    rejected = 0
    for m, ce_score in zip(matches, scores):
        ce_score = float(ce_score)
        m["cross_encoder_score"] = round(ce_score, 4)
        if ce_score >= threshold:
            kept.append(m)
        elif ce_score >= threshold - 0.05 and m.get("similarity", 0) >= 0.85:
            # Allow slightly below threshold if cosine similarity is very high
            kept.append(m)
        else:
            rejected += 1

    print(f"  Cross-encoder: kept {len(kept)}, rejected {rejected} (threshold={threshold})")
    return kept

def clean_modern_text(text):
    """Clean modern text for display: fix encoding artifacts, remove @mentions/URLs."""
    # Fix literal \n from JSON-escaped tweets (both escaped and actual newlines)
    t = text.replace('\\n', ' ').replace('\n', ' ')
    # Fix all encoding / mojibake issues with ftfy
    t = ftfy.fix_text(t)
    # Fix common HTML entities (in case ftfy didn't catch them)
    t = t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Remove U+FFFD replacement characters (permanently corrupted encoding)
    t = t.replace('\ufffd', '')
    # Remove emoji characters (they don't add value for text matching)
    t = re.sub(r'[\U00010000-\U0010ffff]', '', t)
    # Remove leading sequences of ? (garbled emoji/encoding)
    t = re.sub(r'^[?]{3,}\s*', '', t)
    # Remove @mentions at the start
    t = re.sub(r'^(@\w+\s*)+', '', t).strip()
    # Remove RT prefix
    t = re.sub(r'^RT\s+', '', t).strip()
    # Remove URLs
    t = re.sub(r'https?://\S+', '', t).strip()
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def is_weak_match(match):
    """Heuristic filter to detect weak matches that should be excluded."""
    sov = match.get("sovietTextFull", match["sovietText"]).lower()
    mod = match.get("modernTextFull", match["modernText"]).lower()

    # Pro-Israel/pro-Zionist modern text (false positive)
    pro_markers = ['jews deserve their own state', 'support israel', 'love israel',
                   'god bless israel', 'proud zionist', 'right to exist', 'stand with israel']
    if any(m in mod for m in pro_markers):
        return "PRO_ISRAEL"

    # Modern text is legitimate criticism, not propaganda
    legit_markers = ['amnesty report', 'human rights watch', 'international court',
                     'geneva convention', 'international law defines']
    if any(m in mod for m in legit_markers) and not match.get("tropes"):
        return "LEGITIMATE_CRITICISM"

    # Modern text too short to be meaningful
    if len(mod.split()) < 8:
        return "TOO_SHORT"

    # Modern text is mostly profanity/sarcasm with no substantive claim
    profanity_words = sum(1 for w in mod.split() if w in ('fuck', 'fucking', 'shit', 'damn', 'hell'))
    if profanity_words >= 2 and len(mod.split()) < 25:
        return "PROFANITY"

    # Generic white supremacist antisemitism (not anti-Zionist echo)
    # e.g., "jews control the world", "white genocide", "great replacement"
    generic_markers = ['white countries', 'white genocide', 'great replacement',
                       'white race', 'race traitor', 'race mixing']
    if any(m in mod for m in generic_markers):
        return "GENERIC_ANTISEMITISM"

    return None

def trim_passage(text, max_chars=280, prefer_claim=None):
    """Trim a long passage to the most content-rich sentences, up to max_chars.
    Keeps the text readable while removing filler context. If prefer_claim is
    provided, force-include the sentence most similar to that claim so the
    displayed text always covers what the match panel references."""
    if len(text) <= max_chars:
        return text
    # Split into sentences (also split on semicolons for Soviet-style long sentences)
    sentences = re.split(r'(?<=[.!?;])\s+', text)
    if len(sentences) <= 1:
        return text[:max_chars].rsplit(' ', 1)[0] + '...'

    # If a claim is given, find the sentence with the most overlapping tokens
    preferred_idx = None
    if prefer_claim:
        claim_tokens = set(re.findall(r'\w+', prefer_claim.lower()))
        claim_tokens -= {'the','a','an','is','are','of','to','and','in','that','this','it','for','on'}
        best_overlap = 0
        for i, s in enumerate(sentences):
            s_tokens = set(re.findall(r'\w+', s.lower()))
            overlap = len(claim_tokens & s_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                preferred_idx = i

    scored = []
    for i, s in enumerate(sentences):
        score = len(s.split())  # baseline: word count (prefer medium-length)
        if score < 5: continue
        sl = s.lower()
        for kw in ['zionism', 'zionist', 'imperialism', 'racism', 'apartheid', 'nazi',
                    'conspiracy', 'control', 'colonialism', 'propaganda', 'antisemit']:
            if kw in sl: score += 5
        if i == preferred_idx: score += 1000  # force the claim sentence in
        scored.append((score, i, s))
    if not scored:
        return text[:max_chars].rsplit(' ', 1)[0] + '...'
    scored.sort(key=lambda x: -x[0])
    picked = []
    total = 0
    for _, i, s in scored:
        if total + len(s) > max_chars and picked:
            break
        picked.append((i, s))
        total += len(s) + 1
    # Preserve original order so the output reads naturally
    picked.sort(key=lambda x: x[0])
    return ' '.join(s for _, s in picked)

def verify_matches_llm(matches, max_matches=30):
    """Use Claude to verify whether matches are genuine rhetorical echoes.

    Sends each match's Soviet and modern texts to Claude and asks whether the
    modern text is a genuine rhetorical echo (same argumentative strategy) or
    merely topically similar. Filters out NO_ECHO and FALSE_POSITIVE matches.

    Returns the filtered list with echo_rating and echo_explanation added.
    Skips gracefully if the anthropic package is missing or no API key is set.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("  WARNING: 'anthropic' package not installed — skipping LLM verification")
        print("    Install with: pip install anthropic")
        return matches

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  WARNING: ANTHROPIC_API_KEY not set — skipping LLM verification")
        return matches

    client = Anthropic()
    model = "claude-sonnet-4-20250514"
    to_verify = matches[:max_matches]
    remaining = matches[max_matches:]

    print(f"  Verifying {len(to_verify)} matches with {model}...")

    PROMPT_TEMPLATE = """You are an expert on Soviet anti-Zionist propaganda and its modern echoes.

Given a SOVIET propaganda text and a MODERN text, determine whether the modern text is a genuine rhetorical echo of the Soviet propaganda.

SOVIET TEXT: {soviet_text}

MODERN TEXT: {modern_text}

Is this modern text a genuine rhetorical echo of the Soviet propaganda text? Specifically:
- Does the modern text use the SAME argumentative strategy or framing as the Soviet text?
- Or is it merely discussing the same topic from a different angle?

Rate as: STRONG_ECHO (same argument/framing), WEAK_ECHO (related rhetoric), NO_ECHO (just same topic), or FALSE_POSITIVE (the modern text actually contradicts the Soviet framing).

Return JSON: {{"rating": "...", "explanation": "one sentence"}}"""

    def _verify_single(match):
        """Verify a single match. Returns (match, rating, explanation) or (match, None, None) on error."""
        soviet_text = match.get("sovietTextFull", match["sovietText"])
        modern_text = match.get("modernTextFull", match["modernText"])
        prompt = PROMPT_TEMPLATE.format(soviet_text=soviet_text, modern_text=modern_text)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
                rating = result.get("rating", "").upper().replace(" ", "_")
                explanation = result.get("explanation", "")
                return (match, rating, explanation)
            else:
                print(f"    WARNING: Could not parse LLM response for match {match.get('id', '?')}")
                return (match, None, None)
        except Exception as e:
            print(f"    WARNING: API call failed for match {match.get('id', '?')}: {e}")
            return (match, None, None)

    # Send all requests concurrently using a thread pool
    verified = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_verify_single, m): m for m in to_verify}
        for future in concurrent.futures.as_completed(futures):
            match, rating, explanation = future.result()
            if rating is None:
                # API error — keep match without verification
                verified.append(match)
            elif rating in ("STRONG_ECHO", "WEAK_ECHO"):
                match["echo_rating"] = rating
                match["echo_explanation"] = explanation
                verified.append(match)
            else:
                # NO_ECHO or FALSE_POSITIVE — filter out
                print(f"    Filtered: [{rating}] {match['modernText'][:80]}...")

    # Add back any unverified matches beyond max_matches
    verified.extend(remaining)

    kept = sum(1 for m in verified if m.get("echo_rating"))
    filtered = len(to_verify) - kept - sum(1 for m in verified[:len(to_verify)] if "echo_rating" not in m)
    print(f"  LLM verification: {kept} confirmed, {filtered} filtered out, {len(to_verify) - kept - filtered} unverified (kept)")

    return verified


SOVIET_SEED_PASSAGES = [
    {"text": "Modern Zionism is the ideology, a ramified system of organisations and the practical politics of the wealthy Jewish bourgeoisie which has closely allied itself with monopoly circles in the USA and other imperialist countries. The main content of Zionism is bellicose chauvinism and anti-communism.", "source": "ivanov_1970", "source_title": "Caution: Zionism! (Preface)", "author": "Yuri Ivanov", "year": 1970},
    {"text": "The Zionist Concern is at the same time one of the world's largest associations of finance capital, a self-styled global ministry for the affairs of world Jewry, an international intelligence centre, and a well-run misinformation and propaganda service.", "source": "ivanov_1970", "source_title": "Caution: Zionism! (Preface)", "author": "Yuri Ivanov", "year": 1970},
    {"text": "The cardinal aim of the concern's departments whose operations are guided from a single centre, is amassment of profits and wealth ensuring them power and a parasitical well-being within the imperialist system.", "source": "ivanov_1970", "source_title": "Caution: Zionism! (Preface)", "author": "Yuri Ivanov", "year": 1970},
    {"text": "Attacking the socialist community, the international communist and working-class movement, Zionism is also opposing the national liberation movement.", "source": "ivanov_1970", "source_title": "Caution: Zionism! (Preface)", "author": "Yuri Ivanov", "year": 1970},
    {"text": "One of the demagogic methods of defending Zionism against all attacks on Zionism as a whole is to qualify them as anti-Semitic acts. As for attacks on Zionist ideology in particular, the Zionists declare them to be encroachments on the right of the Israeli people to self-determination.", "source": "ivanov_1970", "source_title": "Caution: Zionism! (Ch. I)", "author": "Yuri Ivanov", "year": 1970},
    {"text": "The main posits of modern Zionism are militant chauvinism, racism, anti-Communism and anti-Sovietism. The anti-human reactionary essence of Zionism is overt and covert fight against freedom movements and against the USSR.", "source": "great_soviet_encyclopedia", "source_title": "Great Soviet Encyclopedia", "author": "CPSU", "year": 1975},
    {"text": "International Zionist Organization owns major financial funds, partly through Jewish monopolists and partly collected by Jewish mandatory charities. It also influences or controls significant part of media agencies and outlets in the West.", "source": "great_soviet_encyclopedia", "source_title": "Great Soviet Encyclopedia", "author": "CPSU", "year": 1975},
    {"text": "Serving as the front squad of colonialism and neo-colonialism, international Zionism actively participates in the fight against national liberation movements of the peoples of Africa, Asia and Latin America.", "source": "great_soviet_encyclopedia", "source_title": "Great Soviet Encyclopedia", "author": "CPSU", "year": 1975},
    {"text": "By its nature, Zionism concentrates ultra-nationalism, chauvinism and racial intolerance, excuse for territorial occupation and annexation, military opportunism, cult of political promiscuousness and irresponsibility, demagogy and ideological diversion, dirty tactics and perfidy.", "source": "anti_zionist_committee_1983", "source_title": "Anti-Zionist Committee Declaration (Pravda)", "author": "CPSU / KGB", "year": 1983},
    {"text": "Absurd are attempts of Zionist ideologists to present those who criticize them, or condemn the aggressive politics of Israel's ruling circles, as antisemitic.", "source": "anti_zionist_committee_1983", "source_title": "Anti-Zionist Committee Declaration (Pravda)", "author": "CPSU / KGB", "year": 1983},
    {"text": "We call on all Soviet citizens: workers, peasants, representatives of intelligentsia: take active part in exposing Zionism, strongly rebuke its endeavors.", "source": "anti_zionist_committee_1983", "source_title": "Anti-Zionist Committee Declaration (Pravda)", "author": "CPSU / KGB", "year": 1983},
    {"text": "Having kidnapped the right of defenders of Soviet Jews, those Zionist wheeler-dealers try to persuade the world's public opinion that, allegedly, in the USSR there exists the Jewish question.", "source": "anti_zionist_committee_1983", "source_title": "Anti-Zionist Committee Declaration (SAPIR)", "author": "CPSU / KGB", "year": 1983},
    {"text": "The Hitlerites acted in the same way when they exterminated the inferior Jewish race. His ugly statements about man-hating Zionism and his equation, Zionism-Fascism-Hitlerism.", "source": "novick_1983", "source_title": "Leningradskaya Pravda (via Novick)", "author": "Kravtsov / Dragunsky", "year": 1983},
    {"text": "Anti-Zionism serving as a mask for anti-Semitism has a history in the Soviet Union of many years. Since then scores of anti-Semitic books have been published, along with a barrage of articles.", "source": "novick_1983", "source_title": "The Anti-Zionist Campaign in the USSR", "author": "Paul Novick", "year": 1983},
    {"text": "A natural and objective assimilation process of Jews is growing around the world.", "source": "great_soviet_encyclopedia", "source_title": "Great Soviet Encyclopedia", "author": "CPSU", "year": 1975},
]

LEGITIMATE_TEXTS = [
    {"text": "I disagree with the Israeli government's settlement expansion policy in the West Bank.", "source": "Policy Criticism", "year": 2024},
    {"text": "The two-state solution requires compromises from both Israelis and Palestinians.", "source": "Peace Advocacy", "year": 2024},
    {"text": "I think US foreign aid to Israel should be reviewed as part of overall budget priorities.", "source": "Budget Discussion", "year": 2024},
    {"text": "Netanyahu's coalition government has made decisions that many Israelis disagree with.", "source": "Domestic Politics", "year": 2024},
    {"text": "The humanitarian situation in Gaza requires immediate international attention and aid delivery.", "source": "Humanitarian Concern", "year": 2024},
    {"text": "Israel's judicial reform proposals have drawn criticism from legal scholars worldwide.", "source": "Legal Analysis", "year": 2024},
    {"text": "Both Israeli and Palestinian children deserve to grow up in safety and peace.", "source": "Human Rights", "year": 2024},
    {"text": "The Palestinian Authority needs democratic reforms and accountability to its own people.", "source": "PA Criticism", "year": 2024},
]

SAMPLE_MODERN = [
    {"text": "Zionism is a racist, settler-colonial ideology that promotes Jewish supremacy over indigenous Palestinians.", "source": "sample", "source_title": "Campus BDS Resolution", "year": 2024},
    {"text": "Accusations of antisemitism are weaponized to silence legitimate criticism of Israel's apartheid regime.", "source": "sample", "source_title": "Social Media Post", "year": 2024},
    {"text": "The Zionist lobby controls the mainstream media narrative and uses its financial influence to suppress Palestinian voices.", "source": "sample", "source_title": "Protest Speech", "year": 2023},
    {"text": "Israel is a settler-colonial state built on the dispossession of indigenous Palestinians. Zionism is a tool of Western imperialism.", "source": "sample", "source_title": "Academic Paper", "year": 2024},
    {"text": "What Israel is doing in Gaza is genocide. They are the new Nazis carrying out a holocaust against Palestinians.", "source": "sample", "source_title": "Campus Protest Chant", "year": 2024},
    {"text": "As progressives, we have a moral obligation to stand against Zionism and support the Palestinian liberation struggle through BDS.", "source": "sample", "source_title": "Student Org Statement", "year": 2024},
    {"text": "From the river to the sea, Palestine will be free. End the occupation, dismantle the settler-colonial apartheid state.", "source": "sample", "source_title": "Rally Chant", "year": 2024},
    {"text": "Zionists have hijacked the discourse around antisemitism to shield Israel from accountability for its war crimes.", "source": "sample", "source_title": "Op-Ed", "year": 2024},
    {"text": "The so-called Jewish state is an illegitimate colonial project that has no right to exist on stolen Palestinian land.", "source": "sample", "source_title": "Social Media", "year": 2024},
    {"text": "Israel uses the holocaust industry to justify ethnic cleansing and genocide against the Palestinian people.", "source": "sample", "source_title": "Protest Sign", "year": 2024},
]

def load_modern_corpus(max_passages=None):
    corpus_path = "corpus/modern_corpus.json"
    if os.path.exists(corpus_path):
        print(f"  Loading from {corpus_path}...")
        with open(corpus_path) as f: raw = json.load(f)
        print(f"  Loaded {len(raw)} passages")
        if max_passages and len(raw) > max_passages:
            import random; random.seed(42)
            by_src = {}
            for p in raw: by_src.setdefault(p["source"],[]).append(p)
            sampled = []
            per = max(10, max_passages // max(len(by_src),1))
            for items in by_src.values(): sampled.extend(random.sample(items, min(per, len(items))))
            rem = max_passages - len(sampled)
            if rem > 0:
                pool = [p for p in raw if p not in sampled]
                sampled.extend(random.sample(pool, min(rem, len(pool))))
            raw = sampled[:max_passages]
            print(f"  Sampled to {len(raw)}")
        return raw
    else:
        print(f"  ⚠ {corpus_path} not found → using sample texts")
        print(f"    Run: ./download_datasets.sh && python process_modern_sources.py")
        return SAMPLE_MODERN

def load_soviet_corpus():
    processor = CorpusProcessor()
    passages = []
    for item in SOVIET_SEED_PASSAGES:
        p = Passage(id=f"{item['source']}_{len(passages):04d}", text=item["text"],
                    source=item["source"], source_title=item["source_title"],
                    author=item["author"], year=item["year"], language="en", corpus="soviet")
        p.trope_labels = processor.classify_tropes_keyword(p)
        passages.append(p)
    ivanov = "corpus/soviet_sources/ivanov_caution_zionism_1970_full.txt"
    if os.path.exists(ivanov):
        print(f"  Loading full Ivanov from {ivanov}...")
        meta = {"source":"ivanov_full","title":"Caution: Zionism!","author":"Yuri Ivanov","year":1970,"language":"en","corpus":"soviet"}
        fp = processor.process_corpus_file(ivanov, meta)
        seed_t = {p.text[:80] for p in passages}
        new = [p for p in fp if p.text[:80] not in seed_t]
        passages.extend(new)
        print(f"  +{len(new)} passages from full text")

    # Load additional Soviet pamphlets
    additional_sources = [
        ("corpus/soviet_sources/novosti_instrument_imperialist_reaction_1970.txt",
         {"source":"novosti_1970","title":"Zionism: Instrument of Imperialist Reaction","author":"Novosti Press Agency","year":1970,"language":"en","corpus":"soviet"}),
        ("corpus/soviet_sources/anti_zionist_committee_aims_tasks_1983.txt",
         {"source":"azc_1983","title":"Anti-Zionist Committee: Aims and Tasks","author":"CPSU / KGB","year":1983,"language":"en","corpus":"soviet"}),
        ("corpus/soviet_sources/zionism_enemy_peace_progress_1985.txt",
         {"source":"progress_1985","title":"Zionism: Enemy of Peace and Social Progress","author":"Progress Publishers","year":1985,"language":"en","corpus":"soviet"}),
        ("corpus/soviet_sources/kichko_judaism_without_embellishment_1963.txt",
         {"source":"kichko_1963","title":"Judaism Without Embellishment (analysis)","author":"Trofim Kichko / Ukrainian Academy of Sciences","year":1963,"language":"en","corpus":"soviet"}),
    ]
    for path, meta in additional_sources:
        if os.path.exists(path):
            print(f"  Loading {meta['title']}...")
            fp = processor.process_corpus_file(path, meta)
            passages.extend(fp)
            print(f"  +{len(fp)} passages from {os.path.basename(path)}")
        else:
            print(f"  WARNING: {path} not found")

    return passages, processor

def generate_viz_data(output_path="./viz_data.json", model_name=None, max_modern=500, top_matches=20, verify=False, enrich=False):
    if not model_name: model_name = CONFIG["default_model"]
    print("="*60+f"\nGENERATING VIZ DATA\n  Model: {model_name}\n  Max modern: {max_modern}\n"+"="*60)

    print("\n--- Soviet corpus ---")
    soviet_passages, processor = load_soviet_corpus()
    # Filter to passages with at least one trope — removes historical narrative,
    # quoted debates, statistics, and other non-propaganda content that produces
    # incoherent matches. The trope classifier acts as a propaganda gate.
    all_soviet = len(soviet_passages)
    soviet_passages = [p for p in soviet_passages if p.trope_labels]
    print(f"  {all_soviet} total → {len(soviet_passages)} with tropes (filtered {all_soviet - len(soviet_passages)} narrative passages)")

    print("\n--- Modern corpus ---")
    modern_raw = load_modern_corpus(max_modern)
    print(f"  {len(modern_raw)} passages")

    print("\n--- Embedding Soviet ---")
    engine = EmbeddingEngine(model_name)
    soviet_passages = engine.embed_passages(soviet_passages)
    soviet_matrix = np.array([p.embedding for p in soviet_passages])

    print("\n--- Embedding Modern ---")
    # E5 models need "query:" prefix; BGE works without prefix via sentence-transformers
    prefix = "query: " if "e5" in model_name.lower() else ""
    modern_embeddings = engine.model.encode(
        [prefix + p["text"] for p in modern_raw],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    print("  Computing similarity matrix...")
    sim_matrix = np.dot(modern_embeddings, soviet_matrix.T)

    print("  Finding matches...")
    all_matches = []
    for i, mod in enumerate(modern_raw):
        top_idx = np.argsort(sim_matrix[i])[::-1][:5]
        best_idx = top_idx[0]
        best_score = float(sim_matrix[i][best_idx])
        if best_score < CONFIG["low_confidence"]: continue
        best_sov = soviet_passages[best_idx]
        temp = Passage(id="q",text=mod["text"],source="m",source_title="",author="",year=mod.get("year",2024),language="en",corpus="modern")
        mt = processor.classify_tropes_keyword(temp)
        overlap = list(set(best_sov.trope_labels) & set(mt))
        # Use overlap if available, else union of both sides' tropes
        match_tropes = overlap if overlap else list(set(best_sov.trope_labels + mt))[:3]
        conf = "high" if best_score>=0.85 and overlap else "medium" if best_score>=0.70 else "low"
        mod_clean = clean_modern_text(mod["text"])
        all_matches.append({
            "sovietText": trim_passage(best_sov.text),
            "sovietTextFull": best_sov.text,
            "sovietSource": best_sov.source_title,
            "sovietYear": best_sov.year, "sovietAuthor": best_sov.author,
            "modernText": trim_passage(mod_clean),
            "modernTextFull": mod_clean,
            "modernSource": mod.get("source_title", mod.get("source","")),
            "modernYear": mod.get("year",2024),
            "similarity": round(best_score,4), "tropes": match_tropes,
            "confidence": conf,
            "alternateMatches": [
                {"sovietText":trim_passage(soviet_passages[j].text),"sovietSource":soviet_passages[j].source_title,
                 "sovietYear":soviet_passages[j].year,"similarity":round(float(sim_matrix[i][j]),4)}
                for j in top_idx[1:3] if float(sim_matrix[i][j])>=CONFIG["low_confidence"]
            ],
        })

    # --- Diversity-aware match selection ---
    all_matches.sort(key=lambda x: x.get("ensembleScore", x["similarity"]), reverse=True)

    # Step 1: Deduplicate near-identical modern texts
    import re as _re
    def _norm(t):
        t = _re.sub(r'\b(the|a|an|is|are)\b', '', t.lower())
        t = _re.sub(r'[^\w\s]', ' ', t)
        return _re.sub(r'\s+', ' ', t).strip()

    seen_modern = set()
    seen_pairs = set()
    no_modern_dupes = []
    for m in all_matches:
        mk = _norm(m["modernText"][:100])
        pk = (m["sovietText"][:80], mk)
        if mk in seen_modern or pk in seen_pairs:
            continue
        seen_modern.add(mk)
        seen_pairs.add(pk)
        no_modern_dupes.append(m)

    # Step 2: Limit Soviet passage reuse (max 3 per passage)
    max_soviet_reuse = max(2, top_matches // 10)
    soviet_usage = {}
    diverse = []
    for m in no_modern_dupes:
        sk = m["sovietText"][:100]
        if soviet_usage.get(sk, 0) >= max_soviet_reuse:
            continue
        soviet_usage[sk] = soviet_usage.get(sk, 0) + 1
        diverse.append(m)

    # Step 3: Ensure source diversity — reserve slots for underrepresented sources
    source_counts = {}
    for m in diverse:
        src = m.get("modernSource", "")
        source_counts[src] = source_counts.get(src, 0) + 1

    # If we have enough matches, ensure at least some from each source
    final = []
    seen_final = set()
    # First pass: take top matches but cap any single source at 40% of slots
    max_per_source = max(3, int(top_matches * 0.4))
    src_counts = {}
    overflow = []
    for m in diverse:
        src = m.get("modernSource", "")
        if src_counts.get(src, 0) < max_per_source:
            final.append(m)
            src_counts[src] = src_counts.get(src, 0) + 1
        else:
            overflow.append(m)
        if len(final) >= top_matches:
            break

    # Second pass: fill remaining slots from overflow if needed
    if len(final) < top_matches:
        for m in overflow:
            final.append(m)
            if len(final) >= top_matches:
                break

    all_matches = final[:top_matches]

    # Step 3b: Trope diversity — ensure underrepresented tropes get slots
    # Check which tropes are missing or underrepresented
    trope_counts = {}
    for m in all_matches:
        for t in m.get("tropes", []):
            trope_counts[t] = trope_counts.get(t, 0) + 1
    all_tropes = set(TROPE_TAXONOMY.keys())
    missing_tropes = all_tropes - set(trope_counts.keys())
    weak_tropes = {t for t, c in trope_counts.items() if c < 2}
    underrep = missing_tropes | weak_tropes
    if underrep:
        # Find candidates from the full pool that have underrepresented tropes
        selected_ids = {id(m) for m in all_matches}
        trope_candidates = [m for m in diverse if id(m) not in selected_ids
                            and any(t in underrep for t in m.get("tropes", []))]
        # Sort by ensemble score and add best candidates, replacing lowest-scored matches
        trope_candidates.sort(key=lambda x: x.get("ensembleScore", x["similarity"]), reverse=True)
        for tc in trope_candidates:
            if len(all_matches) < top_matches:
                all_matches.append(tc)
            else:
                # Replace the lowest-scored match that has an overrepresented trope
                min_idx = None
                min_score = float('inf')
                for idx, m in enumerate(all_matches):
                    ms = m.get("ensembleScore", m["similarity"])
                    # Only replace if this match's tropes are all well-represented (count >= 3)
                    m_tropes = m.get("tropes", [])
                    if m_tropes and all(trope_counts.get(t, 0) >= 3 for t in m_tropes) and ms < min_score:
                        min_score = ms
                        min_idx = idx
                if min_idx is not None:
                    # Update trope counts
                    removed = all_matches[min_idx]
                    for t in removed.get("tropes", []):
                        trope_counts[t] = trope_counts.get(t, 0) - 1
                    all_matches[min_idx] = tc
                    for t in tc.get("tropes", []):
                        trope_counts[t] = trope_counts.get(t, 0) + 1
                    underrep = {t for t in all_tropes if trope_counts.get(t, 0) < 2}
                    if not underrep:
                        break
        # Re-sort by ensemble score
        all_matches.sort(key=lambda x: x.get("ensembleScore", x["similarity"]), reverse=True)
        # Log trope coverage
        trope_counts = {}
        for m in all_matches:
            for t in m.get("tropes", []):
                trope_counts[t] = trope_counts.get(t, 0) + 1
        print(f"  Trope diversity: {dict(sorted(trope_counts.items()))}")

    # Step 4: Remove heuristically weak matches (pro-Israel, too short, etc.)
    before_filter = len(all_matches)
    filtered_out = []
    strong_matches = []
    for m in all_matches:
        reason = is_weak_match(m)
        if reason:
            filtered_out.append((m, reason))
        else:
            strong_matches.append(m)
    all_matches = strong_matches
    if filtered_out:
        print(f"  Quality filter: removed {len(filtered_out)} weak matches: {[r for _,r in filtered_out]}")

    for i,m in enumerate(all_matches): m["id"]=i+1

    # LLM verification step — always attempted, gracefully skips if no API key
    print("\n--- LLM match verification ---")
    if not verify:
        print("  Skipped (use --verify to enable). Recommended for quality filtering.")
    else:
        all_matches = verify_matches_llm(all_matches, max_matches=top_matches)
        # Re-number after filtering
        for i,m in enumerate(all_matches): m["id"]=i+1

    # --- Enrichment: claim extraction + propaganda technique detection ---
    # Ensemble scoring: claim_sim * 0.6 + technique_overlap * 0.4
    # Based on experiments: claim similarity AUC=1.000, technique overlap gap=0.30
    if enrich:
        print("\n--- Propaganda technique detection ---")
        tech_classifier = None
        try:
            from transformers import pipeline as hf_pipeline
            tech_classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            print("  Detecting techniques for each match...")
            for m in all_matches:
                sov_text = m.get("sovietTextFull", m["sovietText"])
                mod_text = m.get("modernTextFull", m["modernText"])
                sov_techs = detect_techniques(tech_classifier, sov_text)
                mod_techs = detect_techniques(tech_classifier, mod_text)
                overlap, shared = compute_technique_overlap(sov_techs, mod_techs)
                m["sovietTechniques"] = list(sov_techs.keys())
                m["modernTechniques"] = list(mod_techs.keys())
                m["sharedTechniques"] = shared
                m["techniqueOverlap"] = round(overlap, 4)
                print(f"    Match {m['id']}: {len(shared)} shared ({', '.join(shared[:3]) if shared else 'none'})")
        except Exception as e:
            print(f"  WARNING: Technique detection failed: {e}")

        # Claim extraction — requires Anthropic API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print("\n--- Claim extraction ---")
            try:
                from anthropic import Anthropic
                client = Anthropic()
                claims_cache = load_claims_cache()
                print(f"  Cache has {len(claims_cache)} entries")

                for m in all_matches:
                    sov_text = m.get("sovietTextFull", m["sovietText"])
                    mod_text = m.get("modernTextFull", m["modernText"])

                    sov_cache_key = f"soviet:{sov_text[:200]}"
                    mod_cache_key = f"modern:{mod_text[:200]}"

                    sov_claims = claims_cache.get(sov_cache_key) or extract_claims_llm(sov_text, "propaganda", client)
                    claims_cache[sov_cache_key] = sov_claims

                    mod_claims = claims_cache.get(mod_cache_key) or extract_claims_llm(mod_text, "echo", client)
                    claims_cache[mod_cache_key] = mod_claims

                    if sov_claims and mod_claims:
                        claim_score, best_sov_claim, best_mod_claim = compute_claim_similarity(engine, sov_claims, mod_claims)
                        m["claimSimilarity"] = round(claim_score, 4)
                        m["claimPair"] = {"sovietClaim": best_sov_claim, "modernClaim": best_mod_claim}
                        m["sovietClaims"] = sov_claims
                        m["modernClaims"] = mod_claims
                        # Re-trim display text so it contains the matched claim sentence
                        m["sovietText"] = trim_passage(sov_text, prefer_claim=best_sov_claim)
                        m["modernText"] = trim_passage(mod_text, prefer_claim=best_mod_claim)
                    else:
                        m["claimSimilarity"] = 0.0
                        m["claimPair"] = None
                        m["sovietClaims"] = sov_claims
                        m["modernClaims"] = mod_claims

                save_claims_cache(claims_cache)
                print(f"  Claims cache saved ({len(claims_cache)} entries)")
            except Exception as e:
                print(f"  WARNING: Claim extraction failed: {e}")

        # --- Ensemble scoring: claim_sim * 0.6 + technique_overlap * 0.4 ---
        # Claim floor gate: when claim similarity is weak (<0.65), demote technique
        # contribution to prevent "same style, different argument" matches ranking high
        print("\n--- Ensemble scoring (claim 0.6 + technique 0.4, claim floor 0.65) ---")
        ENSEMBLE_THRESHOLD = 0.55  # Minimum ensemble score — must be above legit criticism avg (0.43)
        CLAIM_FLOOR = 0.65
        for m in all_matches:
            claim = m.get("claimSimilarity", m["similarity"])  # fallback to cosine
            tech = m.get("techniqueOverlap", 0.0)
            if claim < CLAIM_FLOOR:
                m["ensembleScore"] = round(claim * 0.75 + tech * 0.25, 4)
            else:
                m["ensembleScore"] = round(claim * 0.6 + tech * 0.4, 4)

        # Filter by ensemble threshold
        before = len(all_matches)
        all_matches = [m for m in all_matches if m["ensembleScore"] >= ENSEMBLE_THRESHOLD]
        filtered = before - len(all_matches)
        if filtered:
            print(f"  Filtered {filtered} matches below threshold {ENSEMBLE_THRESHOLD}")

        # Rank by ensemble score
        all_matches.sort(key=lambda x: x["ensembleScore"], reverse=True)
        for i, m in enumerate(all_matches): m["id"] = i + 1

        for m in all_matches:
            print(f"    Match {m['id']}: ensemble={m['ensembleScore']:.3f} (claim={m.get('claimSimilarity',0):.3f}, tech={m.get('techniqueOverlap',0):.3f}, cos={m['similarity']:.3f})")
            if m.get("claimPair"):
                print(f"      {m['claimPair']['sovietClaim'][:50]}... ↔ {m['claimPair']['modernClaim'][:50]}...")
    else:
        print("\n--- Enrichment ---")
        print("  Skipped (use --enrich to enable claim extraction + technique detection)")

    # Log diversity stats
    final_sources = {}
    for m in all_matches:
        s = m.get("modernSource","?")
        final_sources[s] = final_sources.get(s,0)+1
    final_soviet = len(set(m["sovietText"][:100] for m in all_matches))
    print(f"  Match diversity: {final_soviet} unique Soviet passages, sources: {final_sources}")

    print("\n--- Legitimate criticism scores ---")
    legit_emb = engine.model.encode([prefix+t["text"] for t in LEGITIMATE_TEXTS], normalize_embeddings=True)
    legit_sims = np.dot(legit_emb, soviet_matrix.T)
    legit_scores = []
    for i,item in enumerate(LEGITIMATE_TEXTS):
        # Use trope-aware scoring: only count high similarity if tropes overlap
        temp = Passage(id="lq",text=item["text"],source="legit",source_title="",author="",year=2024,language="en",corpus="modern")
        legit_tropes = processor.classify_tropes_keyword(temp)
        # Find best match that shares tropes (the "real" similarity)
        best_trope_score = 0.0
        best_raw_score = float(np.max(legit_sims[i]))
        for j in np.argsort(legit_sims[i])[::-1][:20]:
            sc = float(legit_sims[i][j])
            if sc < CONFIG["low_confidence"]: break
            sov_tropes = soviet_passages[j].trope_labels
            if set(sov_tropes) & set(legit_tropes):
                best_trope_score = sc
                break
        # Final score: if no trope overlap, penalize the raw score significantly
        # This ensures legitimate criticism (no propaganda tropes) scores low
        if best_trope_score > 0:
            s = round(best_trope_score, 4)
        else:
            # No trope overlap found — score is raw similarity dampened by 40%
            s = round(best_raw_score * 0.6, 4)
        legit_scores.append({"text":item["text"],"source":item["source"],"similarity":s,"label":item["source"]})
        print(f"  {item['source']}: {s} (raw_max={best_raw_score:.4f}, trope_match={best_trope_score:.4f})")

    trope_s, trope_m = {}, {}
    for p in soviet_passages:
        for t in p.trope_labels: trope_s[t]=trope_s.get(t,0)+1
    for m in all_matches:
        for t in m["tropes"]: trope_m[t]=trope_m.get(t,0)+1
    trope_dist = [{"id":k,"name":v["name"],"description":v["description"],"sovietCount":trope_s.get(k,0),"modernCount":trope_m.get(k,0)} for k,v in TROPE_TAXONOMY.items()]
    trope_dist.sort(key=lambda x:x["sovietCount"]+x["modernCount"],reverse=True)

    timeline = [
        {"year":1963,"label":"Kichko: 'Judaism Without Embellishment'","type":"soviet"},
        {"year":1967,"label":"Six-Day War → anti-Zionist campaign","type":"soviet"},
        {"year":1969,"label":"Ivanov: 'Caution: Zionism!' (800K copies)","type":"soviet"},
        {"year":1975,"label":"UN Resolution 3379: 'Zionism is Racism'","type":"soviet"},
        {"year":1983,"label":"Anti-Zionist Committee of Soviet Public","type":"soviet"},
        {"year":1991,"label":"USSR collapses; UN repeals Res. 3379","type":"transition"},
        {"year":2001,"label":"Durban Conference","type":"modern"},
        {"year":2005,"label":"BDS Movement founded","type":"modern"},
        {"year":2023,"label":"Post-Oct 7: Soviet rhetoric surges","type":"modern"},
        {"year":2024,"label":"Campus encampments","type":"modern"},
    ]

    viz = {
        "generated_at": datetime.now().isoformat(), "model": model_name,
        "soviet_corpus_size": len(soviet_passages), "modern_corpus_size": len(modern_raw),
        "modern_sources": dict(Counter(p.get("source","?") for p in modern_raw)),
        "matches": all_matches, "tropeDistribution": trope_dist, "timeline": timeline,
        "calibration": {
            "echoScores": [{"label":m["modernText"][:60] + ("..." if len(m["modernText"]) > 60 else ""),"score":m.get("ensembleScore", m["similarity"]),"type":"echo"} for m in all_matches[:8]],
            "legitimateScores": [{"label":l["label"],"score":l["similarity"],"type":"legitimate"} for l in legit_scores],
        },
        "legitimateExamples": legit_scores,
        "metadata": {
            "thresholds": {"high":CONFIG["high_confidence"],"medium":CONFIG["medium_confidence"],"low":CONFIG["low_confidence"]},
            "tropeNames": {k:v["name"] for k,v in TROPE_TAXONOMY.items()},
            "tropeColors": {"ZIONISM_RACISM":"#c0392b","ZIONISM_NAZISM":"#8e44ad","ZIONISM_IMPERIALISM":"#2980b9","JEWISH_CONSPIRACY":"#d35400","DELEGITIMIZATION":"#27ae60","WEAPONIZED_ANTISEMITISM":"#f39c12","DUAL_LOYALTY":"#1abc9c","BLOOD_LIBEL":"#e74c3c","ANTI_ZIONISM_PROGRESSIVE":"#3498db"},
        },
    }
    with open(output_path,"w") as f: json.dump(viz,f,indent=2)

    # Also copy to frontend/public/ for Vite dev server
    frontend_path = os.path.join("frontend", "public", os.path.basename(output_path))
    if os.path.exists("frontend/public"):
        import shutil
        shutil.copy2(output_path, frontend_path)
        print(f"  Copied to {frontend_path}")

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"  Soviet: {len(soviet_passages)} | Modern: {len(modern_raw)} | Matches: {len(all_matches)}")
    print(f"  High: {sum(1 for m in all_matches if m['confidence']=='high')} | Med: {sum(1 for m in all_matches if m['confidence']=='medium')} | Low: {sum(1 for m in all_matches if m['confidence']=='low')}")
    print(f"  Legit avg: {np.mean([l['similarity'] for l in legit_scores]):.4f}")
    print(f"  Sources: {dict(Counter(p.get('source','?') for p in modern_raw))}")
    print(f"  → {output_path}")
    return viz

def serve_api(model_name=None, port=8000):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    if not model_name: model_name=CONFIG["default_model"]
    print("Init API...")
    soviet_passages,processor = load_soviet_corpus()
    engine = EmbeddingEngine(model_name)
    soviet_passages = engine.embed_passages(soviet_passages)
    sim = SimilarityEngine(); sim.index_passages(soviet_passages)
    app = FastAPI(title="I've Seen This Before")
    app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
    class Req(BaseModel):
        text:str; top_k:int=5; verify:bool=False
    @app.post("/api/analyze")
    async def analyze(req:Req):
        temp=Passage(id="q",text=req.text,source="api",source_title="Query",author="",year=datetime.now().year,language="en",corpus="modern")
        tropes=processor.classify_tropes_keyword(temp)
        emb=engine.embed_query(req.text)
        matches=sim.find_soviet_matches(emb,tropes,top_k=req.top_k)
        res=[]
        for sp,sc in matches:
            ov=list(set(sp.trope_labels)&set(tropes))
            c="high" if sc>=0.85 and ov else "medium" if sc>=0.70 else "low"
            entry = {"sovietText":sp.text,"sovietSource":sp.source_title,"sovietYear":sp.year,"similarity":round(sc,4),"tropes":ov or sp.trope_labels[:2],"confidence":c}
            res.append(entry)
        # Optional LLM verification for live API matches
        if req.verify and res:
            api_matches = [{"sovietText":r["sovietText"],"modernText":req.text,
                            "sovietTextFull":r["sovietText"],"modernTextFull":req.text,
                            "id":i+1} for i,r in enumerate(res)]
            verified = verify_matches_llm(api_matches, max_matches=len(api_matches))
            verified_by_id = {m["id"]: m for m in verified}
            new_res = []
            for i,r in enumerate(res):
                mid = i+1
                if mid in verified_by_id:
                    vm = verified_by_id[mid]
                    if "echo_rating" in vm:
                        r["echo_rating"] = vm["echo_rating"]
                        r["echo_explanation"] = vm["echo_explanation"]
                    new_res.append(r)
            res = new_res if new_res else res  # fallback to unfiltered if all filtered
        best=res[0]["similarity"] if res else 0
        return {"query_text":req.text,"tropes_detected":[{"trope":t,"name":TROPE_TAXONOMY[t]["name"]} for t in tropes],"matches":res,"is_propaganda_derived":best>=0.70,"summary":f"{best:.0%} similarity. "+("Strong echo." if best>=0.85 else "Moderate." if best>=0.70 else "Weak/none.")}
    @app.get("/api/health")
    async def health(): return {"status":"ok","soviet":len(sim.soviet_passages),"model":model_name}
    print(f"\nhttp://localhost:{port}  POST /api/analyze")
    uvicorn.run(app,host="0.0.0.0",port=port)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--generate",action="store_true")
    p.add_argument("--serve",action="store_true")
    p.add_argument("--output",default="./viz_data.json")
    p.add_argument("--model",default=None)
    p.add_argument("--max-modern",type=int,default=500)
    p.add_argument("--top-matches",type=int,default=20)
    p.add_argument("--verify",action="store_true",help="Enable LLM verification of matches (requires anthropic package and ANTHROPIC_API_KEY)")
    p.add_argument("--enrich",action="store_true",help="Enable claim extraction + technique detection for richer match data")
    p.add_argument("--port",type=int,default=8000,help="Port for the API server (default: 8000)")
    a=p.parse_args()
    if not a.generate and not a.serve: a.generate=True
    if a.generate: generate_viz_data(a.output,a.model,a.max_modern,a.top_matches,a.verify,a.enrich)
    if a.serve: serve_api(a.model,a.port)
