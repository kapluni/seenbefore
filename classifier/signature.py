"""
signature.py — The task definition and initial prompt
=====================================================
A DSPy ``Signature`` is a typed, declarative spec of an LLM task: its
instructions (the docstring) plus input/output fields. DSPy compiles this
into an actual prompt at runtime, and the optimizers in ``tune.py`` rewrite
the instructions and attach few-shot demonstrations to improve it.

The docstring below is the **initial, hand-written prompt** — the starting
point for auto prompt tuning. It encodes:
  * the IHRA working definition framing,
  * the project's non-negotiable rule that legitimate criticism of Israel is
    NOT antisemitism (the "3D test": demonization, double standards,
    delegitimization), and
  * the 9-category Soviet-propaganda trope taxonomy from the parent project,
    so the rationale connects to the project's analytical backbone.
"""

import dspy

from .config import LABEL_POS, LABEL_NEG


# Kept in sync with TROPE_TAXONOMY in embedding_pipeline.py. Reproduced here as
# plain text so the prompt is self-contained and the optimizer can see it.
TROPE_GUIDE = """\
Antisemitic rhetoric frequently maps onto these recurring tropes (a single
post may use several, and these strongly indicate antisemitism):
  1. ZIONISM_RACISM        — Zionism as inherently racist; apartheid/"settler" smears used to deny Jewish self-determination specifically
  2. ZIONISM_NAZISM        — equating Zionists/Israelis with Nazis; "Israel does to Palestinians what Hitler did to Jews"
  3. ZIONISM_IMPERIALISM   — Israel/Jews as the hidden hand of Western imperialism
  4. JEWISH_CONSPIRACY     — Jews/Zionists secretly control media, banks, governments
  5. DELEGITIMIZATION      — denial of Jewish peoplehood/nationhood; "Zionist entity"; Jews are not a real people
  6. WEAPONIZED_ANTISEMITISM — claims that antisemitism accusations are a cynical Zionist trick to silence critics
  7. DUAL_LOYALTY          — Jews/Jewish citizens as disloyal agents of a foreign power
  8. BLOOD_LIBEL           — Jews/Israelis as deliberate child-killers; ritual-murder framing; dehumanization
  9. ANTI_ZIONISM_PROGRESSIVE — antisemitic claims dressed up as progressive anti-racism"""


class ClassifyAntisemitism(dspy.Signature):
    """Classify whether a social-media post is antisemitic.

Use the IHRA working definition: antisemitism is a certain perception of Jews
that may be expressed as hatred toward Jews — rhetorical and physical
manifestations directed toward Jewish or non-Jewish individuals, their
property, community institutions, and (in the case of antisemitic
anti-Zionism) the State of Israel conceived as a Jewish collectivity.

A post is ANTISEMITIC if it does any of the following:
  - expresses hatred, dehumanization, stereotypes, or conspiracy theories about
    Jews as Jews;
  - applies the "3D" test to Israel: DEMONIZATION (e.g. Nazi/blood-libel
    comparisons), DOUBLE STANDARDS (demands of Israel not made of any other
    democracy), or DELEGITIMIZATION (denying Jews alone the right to
    self-determination, "Zionist entity," calling Israel's existence illegitimate);
  - holds Jews collectively responsible for the actions of Israel, or treats
    "Zionist" as a thin substitute for "Jew";
  - uses any of the recurring tropes listed below.

A post is NOT_ANTISEMITIC when it is legitimate political speech, including:
  - criticism of the Israeli government, military, or specific policies of the
    kind that could be leveled at any country;
  - advocacy for Palestinian rights, ceasefires, or humanitarian concern that
    does not demonize, delegitimize, or invoke antisemitic tropes;
  - factual reporting, neutral mention, or counter-speech that rebuts hatred.

This distinction is the single most important judgment: criticizing Israel is
NOT inherently antisemitic. Do not flag legitimate criticism. Equally, do not
excuse genuine antisemitism merely because it is phrased as being "only about
Zionism." Weigh the specific claim, framing, and target.

""" + TROPE_GUIDE

    text: str = dspy.InputField(desc="The social-media post to classify.")

    rationale: str = dspy.OutputField(
        desc="One or two sentences explaining the decision. If antisemitic, name "
             "the relevant trope(s). If not, note why it is legitimate criticism or "
             "neutral speech."
    )
    label: str = dspy.OutputField(
        desc=f"Exactly one of: {LABEL_POS} or {LABEL_NEG}."
    )
