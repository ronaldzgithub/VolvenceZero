"""Optional matplotlib rendering helpers for the Workstream G demos.

matplotlib is an *optional* dependency (``vz-embodiment-ant[viz]``). Every demo
still produces its full JSON data payload without it; only the ``.png`` figures
are skipped. We import it lazily and fail *soft only for the figure*, never for
the data — a missing optional renderer is a documented fallback, not a swallowed
contract error.
"""

from __future__ import annotations

from pathlib import Path


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _pyplot():
    """Return a headless pyplot module, or ``None`` if matplotlib is absent."""

    try:
        import matplotlib
    except ImportError:
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_line_overlay(
    *,
    series: list[dict],
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> Path | None:
    """Plot several (x, y[, style]) series on shared axes. Returns path or None.

    Each series dict: ``{"x": [...], "y": [...], "label": str, "style": str}``.
    """

    plt = _pyplot()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for s in series:
        ax.plot(s["x"], s["y"], s.get("style", "-o"), label=s["label"], linewidth=2.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_trajectory_plot(
    *,
    tracks: list[dict],
    food_markers: list[dict],
    nest: tuple[float, float],
    title: str,
    out_path: Path,
) -> Path | None:
    """Plot 2D ant trajectories with food + nest markers. Returns path or None.

    Each track dict: ``{"xs": [...], "ys": [...], "label": str, "color": str}``.
    Each food marker: ``{"x": float, "y": float, "label": str, "color": str}``.
    """

    plt = _pyplot()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for t in tracks:
        ax.plot(t["xs"], t["ys"], "-", color=t.get("color"), label=t["label"], alpha=0.8)
    for m in food_markers:
        ax.scatter([m["x"]], [m["y"]], marker="*", s=260, color=m.get("color", "goldenrod"),
                   edgecolors="black", zorder=5, label=m["label"])
    ax.scatter([nest[0]], [nest[1]], marker="s", s=140, color="black", zorder=5, label="nest")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
