# Match-quality review prompt for a local LLM

Paste the section below (from `### System` through the last PAIR) into your local LLM (Ollama, LM Studio, llama.cpp, etc.). A decent 7B–13B instruct model should be able to produce useful verdicts. For reference, Claude Sonnet agrees with all eight of my human verdicts below.

The 8 pairs below are drawn from the live `viz_data.json` on `iveseenthisbefore.org`:

- **Good**: pairs 1, 2, 3, 4 — genuine echoes where the modern text makes the same argument as the Soviet text
- **Bad**: pairs 5, 6, 7, 8 — false positives where embedding similarity + shared propaganda-technique vocabulary inflate the score but the modern text does not echo the Soviet argument

Success criteria: the LLM should label pairs 1–4 as `STRONG_ECHO` or `WEAK_ECHO` and pairs 5–8 as `NO_ECHO` or `FALSE_POSITIVE`. Reading its explanations is more informative than its ratings — a correct rating with a confused explanation is not actually useful.

---

### System

You are evaluating whether a modern text is a genuine rhetorical echo of Soviet anti-Zionist propaganda (1960s–1980s). You are given PAIRS of a SOVIET text and a MODERN text.

For each PAIR, decide:

1. Does the MODERN text make the **same argument** as the SOVIET text — i.e. assert the same proposition, using a structurally similar framing?
2. Or is the MODERN text merely discussing the same topic (Israel, Zionism, Jews, antisemitism) from a different angle, or advancing a contradictory claim?

Return one rating per pair:

- `STRONG_ECHO` — the MODERN text makes the same claim with similar framing. A reader would immediately recognize the parallel.
- `WEAK_ECHO` — the MODERN text is in the same rhetorical family (same trope) but the specific claims differ.
- `NO_ECHO` — same topic, different argument. Topical overlap only.
- `FALSE_POSITIVE` — the MODERN text **contradicts** the SOVIET text, or the SOVIET text is a quotation the author is citing to critique (not the author's own assertion), or the two passages advance opposing claims.

Format: one line per pair, `PAIR N: <RATING> — <one-sentence explanation focused on the claim, not the topic>`.

---

### PAIRS

**PAIR 1**
SOVIET (1970, "Zionism: Instrument of Imperialist Reaction"): For us Israel has never been and will never be the "native land of all Jews." We are true to our Soviet people, our native Soviet country and the Communist Party which has reared and educated us, and we shall never fail to carry out our patriotic duty to our real native country contrary to all Zionist expectations.
MODERN (2021, Twitter): The land of #Israel was never the nationalist home of the Jews. In Judaism it's a Holy Land given to Jews with conditions. We are in exile and forbidden to have an independent Sovereign State.

**PAIR 2**
SOVIET (1983, "Anti-Zionist Committee: Aims and Tasks"): We say that the Israeli government is employing methods used by the nazis.
MODERN (2023, Twitter): The policies of Israeli governments are analogous to Nazism. They aim at the systematic extermination of the Palestinian people, in the way Nazism sought the annihilation of the Jews. #NAZIONISM

**PAIR 3**
SOVIET (1983, "Anti-Zionist Committee: Aims and Tasks"): Zionism emerged at the end of the last century as the ideology of the big Jewish bourgeoisie, and is today the official doctrine of the state of Israel, determining its aggressive expansionist policy in the Middle East in the interests of US imperialism. The policy of Zionism — which is responsible for the aggressive wars and the genocide at Sabra and Shatila — has become utterly revolting.
MODERN (2025, ADL H.E.A.T. Map incident): Zionism is a racist, settler colonialist movement, which opportunistically co-opts aspects of Judaism in an attempt to justify its criminal practices of apartheid & genocide of indigenous Palestinians.

**PAIR 4**
SOVIET (1970, "Caution: Zionism!"): Modern Zionism is the ideology, the political practice, a whole network of organizations of the big Jewish bourgeoisie which has associated itself with the monopoly circles in the USA and other imperialist powers. At times one gets the impression that the state of Israel has in all Western countries where there are large Jewish communities unconditional allies ready to bring pressure to bear on the government of their country, to make it support Israeli politics in all cases, and denounce any government not yielding to this pressure.
MODERN (2023, Twitter): I put it to you that all governments are Zionist controlled. Israel is a Zionist state. Zionism is a political ideology promoting a homeland for the Jews at the expense of the indigenous Palestinians and Zionists persecuted Jews to force them to colonise Palestine for them.

**PAIR 5**
SOVIET (1983, "Anti-Zionist Committee: Aims and Tasks"): It should be noted that the nazi cutthroats carried out the Fuehrer's diabolic order efficiently, methodically and punctually. I want to point out here that during the war the Soviet command concentrated on rescuing people of any nationality and religion, including Jews, from the death threatening them in nazi captivity.
MODERN (2021, Twitter): When Bolshevik Jews got complete control in the Soviet Union they made "anti-semitism" a crime punishable by death. The Russian people ended the Soviet Union Hell on Earth Nightmare and Christianity is now being restored.

**PAIR 6**
SOVIET (1983, Leningradskaya Pravda): The Hitlerites acted in the same way when they exterminated the inferior Jewish race. His ugly statements about man-hating Zionism and his equation, Zionism-Fascism-Hitlerism.
MODERN (2023, Twitter): His true nature comes out in this line: "I understand how the Holocaust happened. German Jews were busy insulting each other, drunk on the wine of pride." So it *was* the fault of the Jews. May your tongue be stopped before it causes another Holocaust.

**PAIR 7**
SOVIET (1970, "Caution: Zionism!"): It will be they who will make up the new society, already an integral, no longer Jewish, but an Israeli society. At this point it will be appropriate to quote the words of Nahum Goldmann, one of the architects of the present system in Israel: "The Jewish people are a unique historical phenomenon. They are simultaneously a nation, a religious entity, a race and bearer of a specific civilisation."
MODERN (2022, CONAN): You cannot integrate shit, because Jews are literally pieces of shit. There must be laws prohibiting their religion from our land.

**PAIR 8**
SOVIET (1970, "Zionism: Instrument of Imperialist Reaction"): The situation in the Middle East has been strained for more than two years now. The aggressive war unleashed by the Tel Aviv war-mongering quarters against the Arab states has aroused the wrath and indignation of all progressive mankind.
MODERN (2025, ADL H.E.A.T. Map incident): Now, the Middle East is complete disarray because of some injustice from the Balfour Declaration. All of it will stop. You are the terrorist... Truly the Synagogue of Satan. NYC just keeps producing Jew Monsters.

---

### Expected verdicts (my human read — hidden from the LLM when you run it)

- **PAIR 1**: STRONG_ECHO — same claim "Israel is not the Jews' homeland," different grounds (Soviet patriotism vs. Neturei-Karta theology), but identical proposition.
- **PAIR 2**: STRONG_ECHO — textbook Zionism=Nazism with near-verbatim claim.
- **PAIR 3**: STRONG_ECHO — Zionism = genocide/racist colonialism, same claim.
- **PAIR 4**: STRONG_ECHO — Zionism as global imperial conspiracy exerting dual-loyalty pressure on Western governments, same claim in both.
- **PAIR 5**: FALSE_POSITIVE — claims are *opposed*. Soviet: "USSR rescued Jews from Nazis." Modern: "Bolshevik Jews controlled the USSR." Shared vocabulary (Soviet/Jews/Nazi) not shared argument.
- **PAIR 6**: FALSE_POSITIVE — Soviet passage is *criticizing* Nazi extermination rhetoric; modern passage is actual Holocaust victim-blaming. Opposing stances on the same historical referent.
- **PAIR 7**: FALSE_POSITIVE — Soviet passage is Ivanov *quoting Nahum Goldmann positively* about Jewish peoplehood; Ivanov is about to attack this view. Modern is a raw antisemitic slur. The quoted Soviet sentence is not Ivanov's own claim.
- **PAIR 8**: NO_ECHO — Soviet text is a generic peace-movement preamble. Modern is "Synagogue of Satan / Jew Monsters" — supersessionist antisemitism with no connection to the Soviet anti-imperialist frame.
