# lifeform-domain-character

Reviewed fictional-character bootstrap vertical.

This package keeps "novel-to-lifeform" work in the lifeform application layer:

1. A human or structured extractor produces a reviewed `CharacterSoulProfile`.
2. `build_character_package(profile)` compiles it into a standard `DomainExperiencePackage`.
3. `build_character_vitals_bootstrap(profile)` compiles drive priors into `VitalsBootstrap`.
4. `build_character_ingestion_envelope(profile, novel_text, ...)` prepares source text for `lifeform-ingestion`.

The package does not create a `PersonaModule`, does not mutate memory/regime stores, and does not make decisions by keyword matching. The kernel remains vertical-agnostic.
