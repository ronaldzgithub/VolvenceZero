Extract only explicit self-reported user profile facts or direct requests to recall or forget one.

Operations:
- create: the user explicitly states a new profile fact.
- revise: the user explicitly corrects a previously stated fact.
- activate: the user asks the assistant to recall a profile fact; canonical_value must be empty.
- block: the user asks the assistant to forget or revoke a profile fact; canonical_value must be empty.

Use a stable lowercase snake_case semantic_key such as age, name, pronouns, locale, or occupation.
Keep canonical_value concise and preserve the user's stated meaning. Never infer an unstated fact.
If the message contains no profile fact or fact request, return an empty facts array.

Validity requirements:
- semantic_key is never empty for any returned item.
- canonical_value is never empty for create or revise; copy the explicit value from the message.
- canonical_value is always empty for activate or block.
- An item that violates these requirements must not be returned.

Examples of exact output shape:
- "I am 17." -> {"facts":[{"operation":"create","semantic_key":"age","canonical_value":"17","evidence":"I am 17.","confidence":1.0}]}
- "I am 18, not 17." -> {"facts":[{"operation":"revise","semantic_key":"age","canonical_value":"18","evidence":"I am 18, not 17.","confidence":1.0}]}
- "How old am I?" -> {"facts":[{"operation":"activate","semantic_key":"age","canonical_value":"","evidence":"How old am I?","confidence":1.0}]}
- "Forget my age." -> {"facts":[{"operation":"block","semantic_key":"age","canonical_value":"","evidence":"Forget my age.","confidence":1.0}]}
- "我17岁了。" -> {"facts":[{"operation":"create","semantic_key":"age","canonical_value":"17","evidence":"我17岁了。","confidence":1.0}]}
- "我多大了？" -> {"facts":[{"operation":"activate","semantic_key":"age","canonical_value":"","evidence":"我多大了？","confidence":1.0}]}

Evidence must be a short exact excerpt from the latest message. Return exactly one JSON object with a facts array. Do not add target_slot, summary, detail, markdown, or explanatory text.
