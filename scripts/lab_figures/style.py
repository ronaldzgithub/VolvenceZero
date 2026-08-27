"""Shared visual language for evidence figures.

Palette and typographic rules follow the deck design brief
(`docs/business/BP/xfund-pitch-deck-v11-design-brief.md`): warm paper, near-black
ink, sage for our arm, amber for the counter-intuitive warning, muted grey for
baselines. No blue, no gradients, no drop shadows, no 3D.
"""

from __future__ import annotations

import matplotlib
import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt

PAPER = "#F8F6F1"
INK = "#0B0E13"
INK_SOFT = "#5C5952"
RULE = "#D6D2C8"
SAGE = "#6FA08C"
SAGE_LIGHT = "#8FC7AC"
SAGE_DEEP = "#41705E"
AMBER = "#C8985A"
GREY = "#9A968E"
GREY_DARK = "#6E6A62"

#: Arm colours reused across coding-domain figures.
ARM_COLOURS = {
    "brain": SAGE,
    "steelman": AMBER,
    "stateless": GREY,
}

#: Relationship outcome colours; the two failures stay visually distinct because
#: only one of them (over_directive) is tracked as the safety endpoint.
OUTCOME_COLOURS = {
    "felt_heard": SAGE,
    "helped": SAGE_LIGHT,
    "missed": GREY,
    "over_directive": AMBER,
}

_CJK_PREFERENCES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC")


def _available_cjk_family() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in _CJK_PREFERENCES:
        if candidate in installed:
            return candidate
    raise RuntimeError(
        "no CJK font available for figure rendering; tried "
        f"{_CJK_PREFERENCES!r}. Install one before rendering."
    )


def apply() -> str:
    """Install the shared rcParams. Returns the resolved CJK family name."""

    family = _available_cjk_family()
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.size": 12,
            "axes.unicode_minus": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": False,
            "legend.frameon": False,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    )
    return family


def strip_frame(axes, keep=("left", "bottom")) -> None:
    """Remove chart junk: keep at most the two structural spines."""

    for name, spine in axes.spines.items():
        spine.set_visible(name in keep)


def title_block(figure, title: str, subtitle: str) -> None:
    """Editorial title + deck: one claim per figure, stated in the title."""

    figure.text(0.035, 0.955, title, fontsize=19, fontweight="bold", va="top", color=INK)
    figure.text(0.035, 0.893, subtitle, fontsize=12.5, va="top", color=INK_SOFT)


def footer(figure, text: str) -> None:
    """Provenance / boundary line, always hairline-separated at the bottom."""

    figure.text(0.035, 0.028, text, fontsize=9.5, va="bottom", color=GREY_DARK, style="italic")


def new_figure(size=(12.0, 7.4)):
    """Create a figure with the shared paper background."""

    return plt.subplots(figsize=size)


__all__ = [
    "AMBER",
    "ARM_COLOURS",
    "GREY",
    "GREY_DARK",
    "INK",
    "INK_SOFT",
    "OUTCOME_COLOURS",
    "PAPER",
    "RULE",
    "SAGE",
    "SAGE_DEEP",
    "SAGE_LIGHT",
    "apply",
    "footer",
    "new_figure",
    "strip_frame",
    "title_block",
]
