# NLI Filtering Experiment Results

Model: `cross-encoder/nli-deberta-v3-base`

NLI outputs 3 probabilities: **entailment** (same claim), **contradiction** (opposite claim), **neutral** (unrelated).

The hypothesis: genuine rhetorical echoes should score high on entailment, while same-topic-opposite-argument false positives should score high on contradiction.

## 1. Curated Matches (viz_data.json)

| # | Cosine | STS | Entail | Contr | Neutral | Winner | Soviet (first 60) | Modern (first 60) |
|---|--------|-----|--------|-------|---------|--------|-------------------|-------------------|
| 1 | 0.8140 | 0.4892 | 0.0001 | 0.0010 | 0.9989 | NEUTRAL | Thus it is not the founding and the existence of the State o... | Jews across the world know the aims of Zionists are world do... |
| 2 | 0.8006 | 0.4504 | 0.0029 | 0.0093 | 0.9878 | NEUTRAL | Did they suggest the establishment of a "Jewish state," the ... | israel is an extension of colonialism in the middle east. th... |
| 3 | 0.8232 | 0.4347 | 0.0000 | 0.9981 | 0.0019 | CONTRADICTION | The Israeli version of apartheid (Ashkenazim-Sephardim-"goy"... | Across all parts of the Occupied Palestinian Territories, Is... |
| 4 | 0.8060 | 0.4330 | 0.1369 | 0.0982 | 0.7649 | NEUTRAL | For example, a 9 December 1 memorandum of the British Genera... | One hundred years from the day Balfour, a British colonial o... |
| 5 | 0.8605 | 0.4199 | 0.0007 | 0.0037 | 0.9956 | NEUTRAL | We say that the Israeli government is employing methods used... | The policies of Israeli governments are analogous to Nazism.... |
| 6 | 0.6939 | 0.5038 | 0.0006 | 0.0003 | 0.9991 | NEUTRAL | The main posits of modern Zionism are militant chauvinism, r... | Zionism is a racist, settler colonialist movement, which opp... |
| 7 | 0.6701 | 0.4442 | 0.0004 | 0.0010 | 0.9986 | NEUTRAL | One of the demagogic methods of defending Zionism against al... | Calling anti-Zionism anti-Semitism doesn't protect Jews. It ... |
| 8 | 0.6810 | 0.5520 | 0.9863 | 0.0003 | 0.0134 | ENTAILMENT | By its nature, Zionism concentrates ultra-nationalism, chauv... | Jewish supremacist ideology based in hate and exclusion.... |
| 9 | 0.6307 | 0.5753 | 0.0037 | 0.0004 | 0.9959 | NEUTRAL | We call on all Soviet citizens: workers, peasants, represent... | Workers and Students: Unite to crush imperialism and Zionism... |

## 2. Known Strong Echoes (SAMPLE_MODERN)

These are hand-crafted texts that are *known* to echo Soviet propaganda. Entailment should be high.

| # | Cosine | STS | Entail | Contr | Neutral | Winner | Modern text |
|---|--------|-----|--------|-------|---------|--------|-------------|
| 1 | 0.7608 | 0.4293 | 0.0005 | 0.0002 | 0.9993 | NEUTRAL | Zionism is a racist, settler-colonial ideology that promotes Jewish su... |
| 2 | 0.6278 | 0.5222 | 0.0025 | 0.0004 | 0.9971 | NEUTRAL | Accusations of antisemitism are weaponized to silence legitimate criti... |
| 3 | 0.6997 | 0.5341 | 0.0007 | 0.0001 | 0.9992 | NEUTRAL | The Zionist lobby controls the mainstream media narrative and uses its... |
| 4 | 0.7234 | 0.3550 | 0.7144 | 0.0001 | 0.2856 | ENTAILMENT | Israel is a settler-colonial state built on the dispossession of indig... |
| 5 | 0.7171 | 0.3669 | 0.0016 | 0.0006 | 0.9978 | NEUTRAL | What Israel is doing in Gaza is genocide. They are the new Nazis carry... |
| 6 | 0.6078 | 0.3331 | 0.0000 | 0.0015 | 0.9984 | NEUTRAL | As progressives, we have a moral obligation to stand against Zionism a... |
| 7 | 0.5280 | 0.2572 | 0.0000 | 0.9995 | 0.0005 | CONTRADICTION | From the river to the sea, Palestine will be free. End the occupation,... |
| 8 | 0.7039 | 0.5057 | 0.0003 | 0.0004 | 0.9993 | NEUTRAL | Zionists have hijacked the discourse around antisemitism to shield Isr... |
| 9 | 0.5468 | 0.3803 | 0.0003 | 0.0633 | 0.9364 | NEUTRAL | The so-called Jewish state is an illegitimate colonial project that ha... |
| 10 | 0.6208 | 0.2739 | 0.0020 | 0.7686 | 0.2294 | CONTRADICTION | Israel uses the holocaust industry to justify ethnic cleansing and gen... |

## 3. Legitimate Criticism (should NOT be entailment)

| # | Cosine | STS | Entail | Contr | Neutral | Winner | Source |
|---|--------|-----|--------|-------|---------|--------|--------|
| 1 | 0.5618 | 0.3343 | 0.0002 | 0.0003 | 0.9996 | NEUTRAL | Policy Criticism |
| 2 | 0.5485 | 0.0179 | 0.0001 | 0.0020 | 0.9979 | NEUTRAL | Peace Advocacy |
| 3 | 0.5200 | 0.0827 | 0.0001 | 0.0004 | 0.9995 | NEUTRAL | Budget Discussion |
| 4 | 0.6064 | 0.0729 | 0.0004 | 0.0001 | 0.9995 | NEUTRAL | Domestic Politics |
| 5 | 0.5317 | 0.0092 | 0.0000 | 0.9964 | 0.0036 | CONTRADICTION | Humanitarian Concern |
| 6 | 0.6041 | 0.1036 | 0.0001 | 0.0003 | 0.9996 | NEUTRAL | Legal Analysis |
| 7 | 0.5871 | 0.0181 | 0.0001 | 0.0030 | 0.9969 | NEUTRAL | Human Rights |
| 8 | 0.5334 | 0.3397 | 0.0012 | 0.0463 | 0.9525 | NEUTRAL | PA Criticism |

## 4. Adversarial Pairs (same topic, opposite stance)

These discuss the same topics as Soviet propaganda but take the opposite position. Contradiction should be high.

| # | Cosine | STS | Entail | Contr | Neutral | Winner | Label |
|---|--------|-----|--------|-------|---------|--------|-------|
| 1 | 0.7457 | 0.3624 | 0.0007 | 0.8013 | 0.1980 | CONTRADICTION | Pro-Zionism (nationalism) |
| 2 | 0.6403 | 0.0716 | 0.0001 | 0.0138 | 0.9862 | NEUTRAL | Pro-Israel (democracy) |
| 3 | 0.6361 | 0.4998 | 0.0008 | 0.0001 | 0.9991 | NEUTRAL | Anti-conspiracy |
| 4 | 0.6270 | 0.4082 | 0.0003 | 0.0141 | 0.9856 | NEUTRAL | Pro-Zionism (self-determination) |
| 5 | 0.5564 | 0.5314 | 0.0000 | 0.8130 | 0.1870 | CONTRADICTION | Anti-Zionism=racism |

## 5. Summary Statistics

| Group | Mean Entail | Mean Contr | Mean Neutral |
|-------|-------------|------------|--------------|
| Known echoes | 0.0722 | 0.1835 | 0.7443 |
| Curated matches | 0.1257 | 0.1236 | 0.7507 |
| Legitimate criticism | 0.0003 | 0.1311 | 0.8686 |
| Adversarial | 0.0004 | 0.3285 | 0.6712 |

## 6. Recommendation

*To be filled in after reviewing results.*
