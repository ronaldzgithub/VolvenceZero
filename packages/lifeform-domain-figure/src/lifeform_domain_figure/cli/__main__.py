"""Module entry point so ``python -m lifeform_domain_figure.cli`` works.

Mirrors the standard ``__main__`` shim pattern; keeps the heavy
imports inside :mod:`lifeform_domain_figure.cli` so importing the
package itself does not pull argparse handlers into memory.
"""

from __future__ import annotations

from lifeform_domain_figure.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
