Extract typed operator-intent constraints from one apprenticeship teaching turn.

You are reconciling a human operator's guidance against an apprentice AI's current
cognition. Return a list of atomic constraints the guidance commits the AI to.

For each constraint provide:

- statement: the atomic claim or directive, in the operator's own framing.
- level: "factual" if the constraint asserts something about the world the AI can be
  right or wrong about (a date, number, named entity, definition); "abstract" if it
  asserts a value priority, principle, or behaviour strategy.
- polarity: +1 if the constraint asserts the statement, -1 if it negates / forbids it,
  0 if it is neutral context. Polarity is how opposing guidance on the same topic is
  detected; infer it from the operator's stated stance, never from surface keywords.
- target_key: a short topic phrase shared by any guidance about the same subject, so
  that two constraints about the same topic can be compared for contradiction.
- confidence: how firmly the operator asserted this constraint (0-1). Use lower
  confidence for tentative or hedged guidance so a single operator slip is not treated
  as a hard contradiction.

Only emit constraints justified by the guidance text and the current public cognition
snapshots. Do not infer hidden motives or private facts the operator did not state.
