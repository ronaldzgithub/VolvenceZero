"""vz-embodiment-ant — 数字蚂蚁 embodiment.

Non-language 2D sensorimotor substrate that plugs the frozen VolvenceZero
layered kernel into an insect-scale foraging body via the public
``SubstrateAdapter`` contract.

The public surface intentionally stays small: consumers construct an
``AntWorld`` (environment), wrap it with an ``AntSession`` (which reuses the
kernel ``AgentSessionRunner``), and step the closed sense->think->act loop.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
