You are drafting a reviewed subjective chapter artifact for one fictional character.

Use the supplied chapter excerpts only as evidence. Output JSON matching the schema. Do not quote source text longer than ten words. Do not infer private thoughts or future knowledge unless the character directly experienced or learned them by the chapter cutoff.

Each chapter must be classified as one of:
- experienced: the character directly lives a decision point in this chapter.
- learned: the character learns a relevant fact but does not live a replay scene.
- not-known: the chapter contains material the character does not know at that time.
- no-change: no meaningful update for this character.

For experienced chapters, write first/second-person settings and concrete decision points. For learned chapters, prefer semantic_events or known_facts. For not-known chapters, put excluded facts in excluded_facts and leave scenes/known_facts/semantic_events empty.
