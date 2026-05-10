"""Router skeleton for the OpenAI-compat surface.

Packet 1 ships only the placeholder. Packet 4 will wire in the
``add_openai_routes`` mounting helper that adds
``POST /v1/chat/completions`` to an existing aiohttp ``Application``
built by :func:`lifeform_service.create_app`.

Why this file exists at packet 1 rather than packet 4:

* The boundary contract tests
  (``tests/contracts/test_openai_adapter_*.py``) walk every Python
  file under ``packages/lifeform-openai-compat/src/`` and assert the
  AST imports / call-sites are clean. Having the router file present
  even as a placeholder forces the boundary check to surface
  violations *as the file is written*, instead of waiting for packet
  4 to ship a violation that takes a separate PR to fix.
"""

from __future__ import annotations

# Intentionally no runtime imports beyond stdlib + typing in this
# placeholder. Packet 4 will add ``aiohttp`` and ``lifeform_service``
# imports.
