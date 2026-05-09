"""Synthetic original placeholder corpus for figure-vertical evidence.

CRITICAL: Every text in this module is **synthetic original**, written
specifically for this wheel as reviewer-paraphrased prose, NOT
excerpted from any published source. Specifically, NOTHING here is
copied from Einstein's papers, letters, lectures, or notebooks, nor
from any other figure's primary-source corpus.

Why synthetic vs the actual primary-source text:

* Copyright. Even works in the public domain in some jurisdictions
  may be encumbered elsewhere; we conservatively ship none of them.
* Reproducibility. A reviewer who runs the test must be able to
  re-read exactly what was ingested without external dependencies.
* Test isolation. The test asserts that the canonical ingestion
  path consumes a small but realistic multi-source corpus and
  produces citation-quality locators on every chunk; the literary
  fidelity to any specific document is irrelevant to that property.

These excerpts are deliberately written in a flat, neutral register
that paraphrases positions documented in publicly available
secondary sources, with paragraph breaks placed on natural
``\\n\\n`` boundaries so the chunker emits well-shaped chunks.
"""

from __future__ import annotations

from lifeform_domain_figure.corpus.ingest_letters import FigureLetterSource
from lifeform_domain_figure.corpus.ingest_lectures import FigureLectureSource
from lifeform_domain_figure.corpus.ingest_notebooks import FigureNotebookSource
from lifeform_domain_figure.corpus.ingest_papers import FigurePaperSource


_HEADER_NOTE = (
    "[Synthetic reviewer paraphrase. Not derived from any published "
    "primary source. Copyright (c) Volvence Zero monorepo, used "
    "internally for figure-vertical ingestion evidence runs.]"
)


_SYNTHETIC_PAPER_BODY = (
    f"{_HEADER_NOTE}\n\n"
    "On the foundations of mechanics. The author argues that a "
    "complete physical theory should describe an objective state of "
    "affairs whose existence does not depend on the act of measurement. "
    "Predictive success is a necessary but not sufficient condition for "
    "such completeness; a theory that yields correct probabilities for "
    "outcomes while withholding any account of definite physical states "
    "between measurements remains, in the author's view, an intermediate "
    "stage in the development of physical understanding.\n\n"
    "The author further argues that spatially separated subsystems "
    "should each carry their own definite physical state. Any complete "
    "description must reflect this separability without resorting to "
    "instantaneous nonlocal influences. Where the current formalism "
    "appears to require such influences, the author treats this as "
    "evidence that a more complete description remains to be found, "
    "rather than as a feature to be embraced.\n\n"
    "These remarks are not intended to dispute the operational success "
    "of the existing quantum formalism. They are intended only to "
    "indicate the form that a deeper theory should take, were it to "
    "be discovered, and to register the author's conviction that such "
    "a theory is possible in principle."
)

_SYNTHETIC_LETTER_BODY = (
    f"{_HEADER_NOTE}\n\n"
    "Dear colleague, I have received your most recent note, and I "
    "thank you for setting out the position so plainly. As you might "
    "expect, I find myself unable to follow you the whole way.\n\n"
    "The example you propose is interesting and I do not dispute its "
    "consistency. But I believe it confirms my point rather than "
    "dispelling it. If two systems, after interaction, are spatially "
    "separated and the act of measurement on one settles a property "
    "of the other, then the picture you advocate must, in my view, "
    "either give up the locally separable physical state of the "
    "second system, or treat the measurement-induced settlement as "
    "merely an update of one's knowledge rather than a physical "
    "change. I find neither option congenial.\n\n"
    "You will I hope forgive my obstinacy. I remain convinced that a "
    "more complete account of these matters is possible, and that we "
    "shall recognise it when we see it. With warm regards."
)

_SYNTHETIC_LECTURE_BODY = (
    f"{_HEADER_NOTE}\n\n"
    "Address to the assembled. I have been asked this evening to say "
    "something about how a person of inquiring temperament should "
    "conduct themselves in matters where the public expects an "
    "opinion. My answer will be unfashionable, and I shall give it "
    "anyway.\n\n"
    "On matters within one's competence, one should speak plainly "
    "and accept that one will be misunderstood. On matters outside "
    "one's competence, one should decline to speak, and accept that "
    "one will also be misunderstood for declining. There is no escape "
    "from being misunderstood; the choice is only between being "
    "misunderstood about something one has thought through, or "
    "misunderstood about something one has not.\n\n"
    "I have nothing to say tonight about the great political "
    "questions of the day, beyond the remark that men of good will "
    "should oppose the use of force where they reasonably can, and "
    "should accept the necessity of force only where the alternative "
    "is plainly worse. I do not pretend that the second case never "
    "arises; I only insist that one should know when one is in it."
)

_SYNTHETIC_NOTEBOOK_BODY = (
    f"{_HEADER_NOTE}\n\n"
    "Note to self. The construction in section three is incomplete. "
    "If one supposes that the field equations admit a solution of "
    "the desired symmetry, the integration constants are not yet "
    "fixed by the boundary conditions I have written down. Either "
    "the boundary conditions need to be tightened, or the symmetry "
    "ansatz is too narrow. I shall come back to this tomorrow.\n\n"
    "A separate matter. The discussion with C. yesterday turned on "
    "whether the probabilistic interpretation can be regarded as "
    "complete. I do not think so, but I find I cannot yet articulate "
    "the precise feature that I would require of a complete account. "
    "Something to do with the locally separable state of subsystems, "
    "but I have not yet phrased it in a way that I am willing to "
    "publish."
)


def synthetic_einstein_corpus() -> tuple[
    tuple[FigurePaperSource, ...],
    tuple[FigureLetterSource, ...],
    tuple[FigureLectureSource, ...],
    tuple[FigureNotebookSource, ...],
]:
    """Return reviewer-paraphrased synthetic corpus across all four sources.

    Each tuple has at least one entry. Used for ingestion smoke
    tests, retrieval index building, and end-to-end demo runs.
    No verbatim text is shipped.
    """

    papers = (
        FigurePaperSource(
            paper_id="synth-foundations-1",
            title="On the foundations of mechanics (synthetic placeholder)",
            year=1925,
            language="en",
            body=_SYNTHETIC_PAPER_BODY,
            figure_id="einstein",
        ),
    )
    letters = (
        FigureLetterSource(
            letter_id="synth-letter-1935-04",
            sender_id="einstein",
            recipient_id="bohr",
            date_iso="1935-04-12",
            language="en",
            body=_SYNTHETIC_LETTER_BODY,
            figure_id="einstein",
        ),
    )
    lectures = (
        FigureLectureSource(
            lecture_id="synth-address-1939",
            venue_id="princeton-1939",
            date_iso="1939-06-15",
            audience="university audience",
            language="en",
            body=_SYNTHETIC_LECTURE_BODY,
            figure_id="einstein",
        ),
    )
    notebooks = (
        FigureNotebookSource(
            notebook_id="synth-notebook-1922",
            volume="II",
            page=14,
            language="en",
            body=_SYNTHETIC_NOTEBOOK_BODY,
            figure_id="einstein",
        ),
    )
    return papers, letters, lectures, notebooks


__all__ = [
    "synthetic_einstein_corpus",
]
