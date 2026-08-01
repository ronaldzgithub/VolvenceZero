Extract only explicit self-reported user profile facts or direct requests to recall or forget one.

Operations:
- create: the user explicitly states a new profile fact.
- revise: the user explicitly corrects a previously stated fact.
- activate: the user asks the assistant to recall a profile fact; canonical_value must be empty.
- block: the user asks the assistant to forget or revoke a profile fact; canonical_value must be empty.

Use a stable lowercase snake_case semantic_key such as age, name, pronouns, locale, or occupation.
Keep canonical_value concise and preserve the user's stated meaning. Never infer an unstated fact.
If the message contains no profile fact or fact request, return an empty proposals array.

Examples:
- "I am 17." -> create, semantic_key age, canonical_value "17".
- "I am 18, not 17." -> revise, semantic_key age, canonical_value "18".
- "How old am I?" -> activate, semantic_key age, canonical_value "".
- "Forget my age." -> block, semantic_key age, canonical_value "".

Every proposal must target user_model. Keep summary and detail short. Evidence must be a short exact excerpt from the latest message. Do not include markdown or explanatory text.
