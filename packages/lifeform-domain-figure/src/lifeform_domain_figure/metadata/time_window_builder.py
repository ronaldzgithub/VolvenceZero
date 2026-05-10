"""Pure helper that turns a :class:`FigureLifespan` into time-window hints.

Used by the Wikidata adapter to produce one or more
:class:`TimeWindowHint` records covering the figure's lifespan,
optionally split at reviewer-declared inflection years.
"""

from __future__ import annotations

from lifeform_domain_figure.metadata.records import (
    FigureLifespan,
    MetadataSource,
    TimeWindowHint,
)


def build_time_window_hints_from_lifespan(
    lifespan: FigureLifespan,
    *,
    splits_at_years: tuple[int, ...] = (),
    source: MetadataSource = MetadataSource.WIKIDATA,
    source_id: str = "",
) -> tuple[TimeWindowHint, ...]:
    """Return one or more :class:`TimeWindowHint` records covering the lifespan.

    With ``splits_at_years=()`` the helper returns a single window
    covering ``(birth_year, death_year)`` (or open-ended end if the
    figure is still alive).

    With ``splits_at_years=(y1, y2, ...)`` the helper splits the
    lifespan at every declared year. Splits must be strictly within
    the lifespan; out-of-range splits raise.
    """

    if source_id == "":
        source_id = lifespan.source_id
    end_year = lifespan.death_year if lifespan.death_year is not None else 9999
    if any(y <= lifespan.birth_year or y >= end_year for y in splits_at_years):
        raise ValueError(
            f"build_time_window_hints_from_lifespan: every split must be "
            f"strictly inside ({lifespan.birth_year}, {end_year}); got "
            f"{splits_at_years!r}"
        )
    boundaries = sorted({lifespan.birth_year, *splits_at_years, end_year})
    pairs = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    hints: list[TimeWindowHint] = []
    for index, (start, stop) in enumerate(pairs):
        if index == 0:
            window_id = f"{lifespan.figure_id}:early"
            description = f"Early period {start}-{stop}"
        elif index == len(boundaries) - 2:
            window_id = f"{lifespan.figure_id}:late"
            description = f"Late period {start}-{stop}"
        else:
            window_id = f"{lifespan.figure_id}:mid-{index}"
            description = f"Middle period {start}-{stop}"
        hints.append(
            TimeWindowHint(
                window_id=window_id,
                year_start=start,
                year_end=stop,
                description=description,
                source=source,
                source_id=source_id,
            )
        )
    return tuple(hints)


__all__ = [
    "build_time_window_hints_from_lifespan",
]
