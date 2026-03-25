# Combating Antisemitism & Anti-Zionism: Projects, Volunteer Opportunities & Resources

## About Me (Blurb)

I'm Ilya — an ML engineering manager at Spotify, where I lead teams building large-scale recommendation systems and data infrastructure. I'm hands-on technically but also experienced in scoping projects, coordinating contributors, and driving execution. I emigrated from the USSR in 1991, where I experienced antisemitism firsthand as a child. Though I'm secular, I'm a proud Jew and Zionist, and I want to put both my technical and leadership skills toward combating antisemitism and anti-Zionism — whether that means building tools, organizing volunteer efforts, or showing up in person.

---

## Project Ideas

### 1. "I've Seen This Before" — Soviet-to-Modern Anti-Zionist Language Tracker

**Concept:** Build a semantic similarity engine and public-facing platform that maps Soviet anti-Zionist propaganda language to modern antisemitic/anti-Zionist rhetoric on campuses and social media. Combines Izabella Tabarovsky's qualitative research with quantitative NLP infrastructure.

**Why it matters:** The language used in modern anti-Israel protests — "Zionism is racism," "settler-colonial," "fascist," "apartheid" — was literally engineered by Soviet propagandists in the 1960s-80s. Most people repeating it don't know its origin.

**Format options:**
- Interactive website with side-by-side comparisons (Soviet source → modern echo)
- Semantic similarity tool: input modern text, get matched Soviet-era propaganda
- Short-form video series featuring FSU immigrants drawing the parallels
- Social media campaign with split-screen visuals

**Key collaborator:** Izabella Tabarovsky (author of "Be a Refusenik," senior advisor at Kennan Institute/Wilson Center, leading scholar on Soviet anti-Zionism → modern left antisemitism)

**Related works to build on:**
- Tabarovsky's "Zombie Anti-Zionism" (Tablet Magazine)
- Tabarovsky's "Demonization Blueprints: Soviet Conspiracist Antizionism" (Journal of Contemporary Antisemitism)
- "The Language of Soviet Propaganda" (Quillette)
- ADL's "Contemporary Anti-Zionism's Connections to Soviet Propaganda"
- "Red Terror: How the Soviet Union Shaped the Modern Anti-Zionist Discourse" (Australian Institute of International Affairs)
- "The Anti-Zionist Lexicon" (Times of Israel blog)

**Soviet source texts for corpus:**
- Yuri Ivanov, "Beware: Zionism" (repackaged Protocols)
- Trofim Kichko, "Judaism Without Embellishment" (1963)
- Yevseyev, "Fascism Under a Blue Sky"
- Anti-Zionist Committee of the Soviet Public founding declaration
- Official CPSU definitions of Zionism
- Novosti Press Agency pamphlets (e.g., 76-page pamphlet with 300+ uses of "genocide/terror/racist")
- Works by Bolshakov, Begun, Korneyev, and other state-sponsored anti-Zionist writers

---

### 2. Soviet Jewish Oral History Archive

**Concept:** Record, transcribe (with AI), and make searchable the stories of Soviet Jewish immigrants — focusing on the everyday family experience of antisemitism, the decision to leave, and what Zionism meant to them. Time-sensitive: the generation that made the decision to leave is aging.

**Why it's unique:** Existing archives (AJHS, Yeshiva University) focus on the *movement* — famous refuseniks and American activists. The stories of ordinary families who simply lived through it as children haven't been captured at scale.

**Local angle:** Large Russian-speaking Jewish community in Brookline/Newton area near Canton, MA. Some Yeshiva University oral history interviews were recorded in Waltham, MA.

**Existing archives to partner with (not duplicate):**
- Archive of the American Soviet Jewry Movement (AJHS/Center for Jewish History) — 1,400+ linear feet, 78,801 digitized images, 544 hours of audio
- Yeshiva University Soviet Jewry Oral History Project — video interviews with movement activists
- Harvard Judaica Collection — 5.5 million digital images
- East View Judaica Digital Collections — "Jewish Emigration from the USSR" (1,466 files, 30,939 pages)

---

### 3. Radicalization Pathway Mapper

**Concept:** Reverse-engineer how recommendation algorithms funnel people from mainstream content into antisemitic rabbit holes. Publish findings as research and advocacy material for policymakers.

**Why me:** Very few people in the anti-hate space understand how recommendation systems actually work from the inside. This is a unique contribution from someone with RecSys expertise at a major platform.

---

### 4. Counter-Narrative Recommendation Engine

**Concept:** Instead of just flagging bad content, build a system that helps pro-Israel and Jewish educational content find the right audiences. A tool for creators and orgs to optimize content distribution.

---

### 5. Antisemitism Trend Dashboard

**Concept:** Public, real-time dashboard visualizing how antisemitic narratives spread across platforms — what triggers spikes, which tropes are rising/falling, platform response rates. "Spotify Wrapped for antisemitism."

---

### 6. School Curriculum Watchdog Tool

**Concept:** NLP tool that helps parents and school boards analyze curricula, textbooks, and supplementary materials for antisemitic or anti-Zionist bias. Scans against IHRA definition and known problematic framings.

**Relevance:** ADL, Brandeis Center, and StandWithUs recently expanded their K-12 antisemitism helpline to Massachusetts.

---

### 7. Debate Prep AI

**Concept:** Training tool for pro-Israel advocates (especially students) that simulates hostile Q&A using an LLM, helps practice responses, gives feedback. Flight simulator for difficult Israel conversations.

**Potential partners:** Hillel, StandWithUs campus programs, Israel on Campus Coalition

---

### 8. Hack the Hate Hackathon (Boston Edition)

**Concept:** Organize a local hackathon focused on building tools against antisemitism. Precedent: similar event held at Microsoft Tel Aviv offices in 2023 ("Hack the Hate" organized by Generative AI for Good).

**Potential co-sponsors:** CJP, ADL New England, local tech companies

---

### 9. Tech Professionals for Israel Network

**Concept:** Slack community or meetup group for tech professionals in Boston who want to volunteer skills against antisemitism. Be the connector between talent and orgs that need it.

---

## Datasets & Data Sources

### Labeled Training Data (for classifiers)

| Dataset | Size | Description | Access |
|---------|------|-------------|--------|
| ISCA Twitter Dataset | 6,941 tweets | Labeled antisemitic/non-antisemitic using IHRA definition (2019-2021), 18% antisemitic | doi.org/10.5281/zenodo.7932888 (CC 4.0) |
| UC Berkeley Measuring Hate Speech | 39,565 comments | 10 ordinal labels, 8 target groups incl. religion, continuous severity score | huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech |
| CONAN Counter-Narratives | 5,003 pairs + 3,000 dialogues | Hate speech / counter-narrative pairs covering Jews and other targets | github.com/marcoguerini/CONAN |
| Hatespeechdata.com | Meta-catalogue | Links to dozens of downloadable hate speech corpora | hatespeechdata.com |

### Incident & Trend Data

| Source | Description | Access |
|--------|-------------|--------|
| ADL H.E.A.T. Map | Interactive map of hate/extremism/antisemitism incidents, updated monthly, CSV download available with incident type and Israel-related filters | adl.org/resources/tools-to-track-hate/heat-map |
| ADL Global A.T.L.A.S. | Global antisemitism trends, country-by-country analysis | Via ADL tools |
| ADL Global 100 Index | Survey data from 103 countries on 11 antisemitic stereotypes | adl.org/adl-global-100-index-antisemitism |
| FBI Crime Data Explorer | Raw hate crime data since 1991; 1,938 anti-Jewish incidents in 2024 (70% of all religion-based hate crimes) | cde.ucr.cjis.gov |
| CAM Antisemitism Research Center | 12,000+ tracked incidents over 5 years | Contact CAM for data access |

### Real-Time Monitoring Platforms

| Platform | Description | Access |
|----------|-------------|--------|
| CyberWell Open Database | 10,000+ vetted antisemitic social media posts, categorized by IHRA definition, in English and Arabic | app.cyberwell.org (register; researcher access available) |
| CyberWell uses Bright Data web scrapers for collection | Partners with Meta, TikTok, YouTube | cyberwell.org |
| FOA AI Monitoring Tool | AI-powered tool scanning social media for antisemitic content, built with Code for Israel | foantisemitism.org |
| ADL Tracker | Near real-time feed of antisemitism developments across US and abroad | adl.org/adl-tracker |

### Academic Research

| Resource | Description |
|----------|-------------|
| "Decoding Antisemitism" (Frontiers in Communication, 2025) | Evaluates transformer-based models and LLMs for antisemitism detection; includes annotation methodology |
| "Subverting the Jewtocracy" (arXiv:2104.05947) | Multimodal antisemitism detection using text + images |
| Ozalp et al. Twitter study | 31M tweets dataset on Jewish identity topics (2015-2016) |
| IHRA annotated dataset (Indiana University ISCA) | Tweet-level annotations using IHRA working definition |

---

## Volunteer Opportunities

### Online Organizations (Outreach Emails Drafted)

| Organization | What They Do | How I Can Help | Contact |
|--------------|-------------|----------------|---------|
| **Fighting Online Antisemitism (FOA)** | AI-powered monitoring and volunteer reporting of antisemitic social media content | Technical contribution to AI monitoring tool, organize volunteer tech efforts, join reporting network | foantisemitism.org/join-us/ |
| **CyberWell** | Open database of antisemitic content, partners with major platforms on hate speech enforcement | ML model improvement, data analysis, help organize volunteer engineering projects | cyberwell.org |
| **Combat Antisemitism Movement (CAM)** | Coalition of 850+ interfaith orgs, 5M+ activists, Report It tool, Antisemitism Research Center | Technical skills for digital tools, help organize projects, join activist network | combatantisemitism.org |
| **StopAntisemitism** | Grassroots org that exposes antisemites, grades campuses, drives consequences | Technical support for research/monitoring, campus grading data, social media | stopantisemitism.org |

### Local In-Person (Canton, MA / Greater Boston)

| Organization | What They Do | How I Can Help | Details |
|--------------|-------------|----------------|---------|
| **B'nai Tikvah, Canton** | CJP partner synagogue | Connect with social action committee for advocacy and Israel solidarity events | Local to Canton |
| **JF&CS Family Table (Canton)** | Kosher food pantry with Sunday morning distributions in Canton | Pack and deliver groceries; kids can volunteer too | jfcsboston.org |
| **Jewish Big Brothers Big Sisters (JBBBS)** | One-to-one mentoring for children and adults with disabilities | Become a Big Brother; actively recruiting near Canton | jbbbs.org |
| **ADL New England** | Anti-bias education in 120+ NE schools, security training, advocacy | Volunteer, attend events, join young leadership board | newengland.adl.org |
| **CJP Center for Combating Antisemitism** | $1.7M in grants for anti-antisemitism work in Boston area; campus, K-12, civic focus | Join Hineni Volunteer Network (skills-based matching), WhatsApp group, monthly newsletter | ma.cjp.org/center-for-combating-antisemitism |
| **CJP Communal Security Initiative** | Collaborated with 250+ Jewish orgs, $1.5M+ in security grants and training | Security volunteer, training participation | Via CJP |
| **AIPAC** | 6M+ members, bipartisan pro-Israel advocacy, lobby days with congressional reps | Attend local advocacy events, lobby days | aipac.org |
| **StandWithUs** | Israel education org with campus programs, legal department, community resources | Volunteer at events, provide resources in community | standwithus.com |
| **School Board Engagement** | Showing up when curriculum, incident response, or DEI frameworks are discussed | Attend Canton school board meetings; ADL/Brandeis Center/StandWithUs K-12 helpline now covers MA | Local |

### Other Relevant Organizations

| Organization | Focus |
|--------------|-------|
| Blue Square Alliance (Robert Kraft, est. 2025) | Inspiring Americans to stand up to Jewish hate and all hate |
| Hillel / Israel on Campus Coalition | Campus Jewish life, pro-Israel advocacy, grants for campus initiatives |
| WJC TecHRI (HO:PE project) | Browser extension and mobile app for reporting antisemitic content |

---

## Key People to Connect With

| Person | Role | Relevance |
|--------|------|-----------|
| **Izabella Tabarovsky** | Senior advisor, Kennan Institute/Wilson Center; author of "Be a Refusenik" | Leading scholar on Soviet anti-Zionism → modern antisemitism; potential collaborator on "I've Seen This Before" project |
| **Tal-Or Cohen Montemayor** | Founder & ED, CyberWell | Leads the open database and platform partnerships |
| **Anat Zalmanson-Kuznetsov** | Filmmaker, creator of The Refusenik Project | Daughter of refuseniks; educational initiative at Bar Ilan University |
| **Dara (CJP)** | Director of Volunteer Mobilization, CJP Hineni Network | Skills-based volunteer matching for Greater Boston Jewish orgs |
| **Jeremy Brick** | ADL New England Young Leadership contact | Entry point for ADL NE involvement |

---

## Key References & Reading

### Books
- **"Be a Refusenik: A Jewish Student's Survival Guide"** — Izabella Tabarovsky (foreword by Natan Sharansky). Pairs Soviet refuseniks with modern campus activists chapter by chapter.

### Articles
- "Zombie Anti-Zionism" — Tabarovsky, Tablet Magazine
- "The Language of Soviet Propaganda" — Quillette
- "Red Terror: How the Soviet Union Shaped the Modern Anti-Zionist Discourse" — Australian Institute of International Affairs
- "The Anti-Zionist Lexicon: How Soviet Propaganda Became Campus Orthodoxy" — Kile Jones, Times of Israel
- "Contemporary Anti-Zionism's Connections to Soviet Propaganda" — ADL
- "Demonization Blueprints: Soviet Conspiracist Antizionism" — Tabarovsky, Journal of Contemporary Antisemitism

### Educational Resources
- ADL "Antisemitism Uncovered: A Guide to Old Myths in a New Era" — antisemitism.adl.org
- The Refusenik Project (Lookstein Center, Bar Ilan University) — free lesson plans on Soviet Jewry
- USHMM antisemitism resources — ushmm.org/antisemitism

---

*Document created: March 2026*
*Next step: Technical architecture plan for "I've Seen This Before" project*
