"""Four stable renderers for an ``AffordanceRegistry``.

Every renderer is a pure function of a descriptor tuple (often
``registry.all_descriptors()``). They do not call back into the
registry and they do not read any kernel state \u2014 hand them a
tuple, get a string / dict out.

* ``render_markdown`` \u2014 LLM-facing prose, human-readable, suited
  for embedding in a system prompt.
* ``render_openai_tools`` \u2014 list of OpenAI-tools-style JSON objects
  suitable for passing to a ``chat.completions.create`` call.
* ``render_catalog_json`` \u2014 full catalog as a JSON-serialisable
  dict; for external dashboards / management UIs.
* ``render_compact_list`` \u2014 ``[name, display_name]`` pairs for
  terse inline reference.

All renderers produce deterministic output ordered by the order of
``descriptors`` (usually registration order).
"""

from lifeform_affordance.renderers.catalog_json import render_catalog_json
from lifeform_affordance.renderers.compact_list import render_compact_list
from lifeform_affordance.renderers.markdown import render_markdown
from lifeform_affordance.renderers.openai_tools import render_openai_tools

__all__ = [
    "render_catalog_json",
    "render_compact_list",
    "render_markdown",
    "render_openai_tools",
]
