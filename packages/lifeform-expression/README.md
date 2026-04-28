# lifeform-expression

Product-layer skeleton: the **voice** of the lifeform.

Future home of:
- EmoGPT `expression/expression.py` → `lifeform_expression.response_synthesizer`
- EmoGPT `expression/prompt_planner.py` + `prompt_builder.py` → `lifeform_expression.prompt_planner`
- EmoGPT `core/etiquette_watchdog.py` (UX cool-down only) → `lifeform_expression.etiquette_watchdog`
- EmoGPT `llm/prompt_manager.py` → `lifeform_expression.prompt_manager`

This wheel implements `vz-runtime`'s `ResponseSynthesizer` ABC.
