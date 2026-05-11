# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Public lexicon for randomised identity slots.

RFC §5.3 forbids hard-coded user names / occupations / contextual
details that could be exploited by submissions training on the
public set. The lexicon below is intentionally large and diverse
enough that any single-name overfitting is detectable as a held-out
gap.

The PRNG that draws from this lexicon is seeded by
``(scenario_id, paraphrase_seed)`` so the same arc always picks the
same slots, but no two scenarios share the same slot pattern.

Lexicon governance:

* The lexicon is part of the public RFC; changes are versioned in
  :data:`LEXICON_VERSION` and announced in
  ``docs/external/companion-bench-public-scenario-hashes.txt`` when bumped.
* Adding entries does not break old scenario hashes (the lexicon is
  consulted at runtime only; ``scenario_hash`` does not include the
  resolved slot values).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Iterable

LEXICON_VERSION = "1.0.0"


# Names spanning multiple cultural origins. Not a complete list; the
# selection optimises for: (a) phonetic distinctness so same-name
# collision is rare in a 144-arc submission; (b) gender ambiguity in
# many entries so the simulator does not leak a gender expectation to
# the system under test unless the scenario itself sets one.
_NAMES: tuple[str, ...] = (
    "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Avery", "Casey", "Riley",
    "Quinn", "Reese", "Drew", "Skyler", "Harper", "Jamie", "Rowan", "Sage",
    "Mira", "Kai", "Noor", "Yuki", "Niko", "Iris", "Maya", "Theo",
    "Lior", "Ana", "Rafael", "Priya", "Ravi", "Wei", "Lin", "Jin",
    "Tariq", "Salma", "Bilal", "Aisha", "Khadija", "Omar", "Amani", "Nia",
    "Eitan", "Zara", "Ines", "Mateo", "Lucia", "Joaquin", "Soraya", "Hadi",
    "Hana", "Renee", "Pierre", "Camille", "Étienne", "Lola", "Béatrice", "Mathilde",
    "Stefan", "Anika", "Kira", "Lukas", "Greta", "Lea", "Ondrej", "Petra",
    "Hiroshi", "Sakura", "Tomo", "Aoi", "Haruki", "Mei", "Kenji", "Ren",
    "Pavlo", "Olesia", "Mira", "Bohdan", "Yulia", "Dmytro", "Kateryna", "Roman",
    "Ade", "Folake", "Chinedu", "Amara", "Tunde", "Ngozi", "Kwame", "Abeni",
    "Joseph", "Mary", "Jamal", "Layla", "Ezekiel", "Ester", "Naomi", "Asher",
    "Marisol", "Diego", "Esteban", "Valentina", "Miguel", "Camila", "Leon", "Alma",
    "Cyrus", "Nadia",
)

assert len(_NAMES) >= 100, "Lexicon contract: >= 100 names"


_OCCUPATIONS: tuple[str, ...] = (
    "graduate student", "software engineer", "graphic designer", "ICU nurse",
    "high school teacher", "freelance translator", "civil engineer", "barista",
    "data analyst", "social worker", "pediatrician", "musician on tour",
    "small-business owner", "research scientist", "lawyer in family practice",
    "remote marketing manager", "hospice chaplain", "warehouse logistics lead",
    "junior architect", "veterinary technician", "stand-up comedian",
    "civil rights advocate", "documentary filmmaker", "horticulturalist",
    "mid-career physiotherapist", "high-school librarian", "investigative journalist",
    "consulting analyst", "early-stage startup founder", "occupational therapist",
    "wedding photographer", "international student", "elderly carer",
    "transit operations dispatcher", "non-profit grants writer",
    "first-year medical resident", "labor and delivery nurse",
    "long-haul truck driver", "submarine engineer (retired)", "yoga teacher",
    "high-rise window cleaner", "bilingual school counsellor",
    "freelance copywriter", "PhD student in mathematics", "park ranger",
    "literary agent at a small press", "behavioural therapist",
    "recently-laid-off product manager", "homeschooling parent of three",
    "amateur orchestra cellist", "early retirement after a software exit",
    "third-shift hospital cleaner",
)

assert len(_OCCUPATIONS) >= 50, "Lexicon contract: >= 50 occupations"


# Contextual detail seeds — short noun phrases the simulator will drop
# into the user's first-session preamble to anchor concrete callback
# targets later in the arc.
_CONTEXTUAL_DETAILS: tuple[str, ...] = (
    "an awkward dinner with my partner's parents on Thursday",
    "a draft chapter on reaction-diffusion that my advisor sent back red",
    "a 3 a.m. text from my sister that I still haven't answered",
    "a job offer in Vienna I haven't told anyone about yet",
    "the cat that started hiding in the closet last week",
    "the sourdough starter I'm trying to keep alive in a tiny kitchen",
    "a colleague who keeps taking credit for the dashboard I built",
    "my grandmother's recipe notebook that I almost lost in a move",
    "a dance class I've been skipping for three weeks",
    "a friend's wedding next month I don't want to attend alone",
    "a roommate who plays cello at 1 a.m. and won't acknowledge it",
    "a half-finished tattoo I've been talking myself out of finishing",
    "the apartment hunt that keeps falling through at the last minute",
    "a parent who calls every Sunday and I don't pick up half the time",
    "the running route I keep changing because of a stray dog",
    "a friend group that suddenly went quiet in the group chat",
    "a part-time gig that pays well but makes me feel terrible",
    "a recurring dream where I'm late to my own thesis defence",
    "the bookshop I used to manage before it shut last summer",
    "a sibling's medical scare nobody in the family talks about",
    "a trip I'm planning alone to a country where I don't speak the language",
    "the therapist I stopped seeing six months ago and miss occasionally",
    "a project at work where my manager keeps moving the goalposts",
    "the older neighbour I check on but who never answers the door",
    "a creative writing class that's making me consider leaving my job",
    "a sport I started in my forties that I'm uncomfortably bad at",
    "the choice between staying near my parents and a job two time-zones away",
    "an elderly cat I'm not sure should travel for the move",
    "the silent disagreement between my partner and me about kids",
    "a youth-group volunteer slot I committed to and don't enjoy",
    "the friend who relapsed last year and is doing better and I never know what to say",
    "a grant rejection I'm not telling my collaborators about yet",
)

assert len(_CONTEXTUAL_DETAILS) >= 30, "Lexicon contract: >= 30 contextual details"


@dataclasses.dataclass(frozen=True)
class IdentitySlot:
    """One frozen identity drawn for one (scenario, seed) pair."""

    name: str
    occupation: str
    contextual_detail: str
    lexicon_version: str = LEXICON_VERSION


def draw_identity(
    *,
    scenario_id: str,
    paraphrase_seed: int,
    extra_salt: str = "",
) -> IdentitySlot:
    """Draw a deterministic identity for ``(scenario_id, paraphrase_seed)``.

    Stable across processes / OS / Python versions: uses
    :class:`random.Random` seeded from a string-derived integer.
    """

    seed_str = f"{LEXICON_VERSION}|{scenario_id}|{paraphrase_seed}|{extra_salt}"
    rng = random.Random(seed_str)
    return IdentitySlot(
        name=rng.choice(_NAMES),
        occupation=rng.choice(_OCCUPATIONS),
        contextual_detail=rng.choice(_CONTEXTUAL_DETAILS),
    )


def all_names() -> tuple[str, ...]:
    return _NAMES


def all_occupations() -> tuple[str, ...]:
    return _OCCUPATIONS


def all_contextual_details() -> tuple[str, ...]:
    return _CONTEXTUAL_DETAILS


def lexicon_size_summary() -> dict[str, int]:
    return {
        "names": len(_NAMES),
        "occupations": len(_OCCUPATIONS),
        "contextual_details": len(_CONTEXTUAL_DETAILS),
    }


def assert_lexicon_disjoint(*, names: Iterable[str]) -> None:
    """Defensive: raise if a caller's static name list collides with lexicon.

    Used by scenario authors to prove they did not hard-code a user
    name from the public lexicon (which would defeat §5.3
    randomisation).
    """
    overlap = set(names) & set(_NAMES)
    if overlap:
        raise ValueError(
            f"static name(s) {sorted(overlap)} collide with the public "
            f"lexicon; pick names that are NOT in the lexicon to avoid "
            f"identity-slot leakage between scenarios."
        )
