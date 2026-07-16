"""Static guard for the semantic embedding SSOT (closes ``known-debts.md`` #3).

Two invariants:

1. **Runtime identity**: the three previously divergent forks
   (``application.scoring_helpers``, ``dual_track.core``,
   ``evaluation.semantic_readouts``) all bind their public name to the
   canonical implementation in :mod:`volvence_zero.semantic_embedding`.

2. **No new forks**: AST scan over ``packages/**/*.py`` rejects any new
   ``def _semantic_embedding`` outside an explicit allow-list for two
   pre-existing, *signature-distinct* helpers that
   ``known-debts.md`` #3 deliberately did not target:

   * ``vz-memory/memory/retrieval.py`` (``text=, tags=, dim=6`` signature
     — keyword-only, accepts tags, defaults to dim 6)
   * ``vz-substrate/substrate/adapter.py`` (``dim=256`` — used to
     project residual-stream surfaces, not goal text)

   Both are tracked separately (``research/core-author-paper-assessment``
   #5) and may be unified once the real embedding head is wired.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"


@pytest.fixture(autouse=True)
def _reset_semantic_embedding_backend():
    """Keep the process-global embedding backend clean around each test.

    #91 added an injectable backend seam. Tests that install a fake backend
    must not leak it into other tests; default is the stub fallback.
    """
    from volvence_zero.semantic_embedding import reset_semantic_embedding_backend

    reset_semantic_embedding_backend()
    yield
    reset_semantic_embedding_backend()


# Files that own a *different* ``_semantic_embedding`` signature and
# therefore are not part of known-debts #3. New entries here should be
# rare and require a corresponding update to ``docs/known-debts.md``.
_ALLOWED_FORK_PATHS: frozenset[pathlib.Path] = frozenset(
    {
        PACKAGES_ROOT / "vz-memory" / "src" / "volvence_zero" / "memory" / "retrieval.py",
        PACKAGES_ROOT / "vz-substrate" / "src" / "volvence_zero" / "substrate" / "adapter.py",
    }
)


def test_forks_share_canonical_entrypoints() -> None:
    """SSOT snapshot (known-debts #3 + #91).

    All embedding consumers bind to the canonical
    :mod:`volvence_zero.semantic_embedding` module:

    * ``tokens`` everywhere are the canonical stub helpers (tokens are a
      lexical notion, not an embedding-space one).
    * ``application.scoring_helpers`` / ``dual_track`` / ``evaluation``
      all route embedding through the #91 seam ``semantic_embedding``
      (stub fallback when no backend is set), so a real substrate
      backend is honored when injected — including the lazily computed
      prototype vectors (M1 / #91 follow-up).
    """
    from volvence_zero.semantic_embedding import (
        semantic_cosine,
        semantic_embedding as seam_embedding,
        stub_cosine_similarity,
        stub_semantic_tokens,
    )
    from volvence_zero.application.scoring_helpers import (
        cosine_similarity,
        semantic_embedding,
        semantic_tokens,
    )
    from volvence_zero.dual_track.core import (
        _cosine_similarity as dual_track_cosine,
        _semantic_embedding as dual_track_embedding,
        _semantic_tokens as dual_track_tokens,
    )
    from volvence_zero.evaluation.semantic_readouts import (
        _cosine_similarity as evaluation_cosine,
        _semantic_embedding as evaluation_embedding,
        _semantic_tokens as evaluation_tokens,
    )

    assert semantic_embedding is seam_embedding
    assert semantic_tokens is stub_semantic_tokens
    assert cosine_similarity is semantic_cosine

    assert dual_track_embedding is seam_embedding
    assert dual_track_tokens is stub_semantic_tokens
    assert dual_track_cosine is stub_cosine_similarity

    assert evaluation_embedding is seam_embedding
    assert evaluation_tokens is stub_semantic_tokens
    assert evaluation_cosine is stub_cosine_similarity


def test_seam_defaults_to_stub_when_no_backend() -> None:
    """#91: with no backend installed, ``semantic_embedding`` == stub."""
    from volvence_zero.semantic_embedding import (
        semantic_embedding,
        stub_semantic_embedding,
    )

    for text in ("hello world", "决定优先级", ""):
        assert semantic_embedding(text, dim=8) == stub_semantic_embedding(text, dim=8)


def test_injected_backend_routes_and_reset_restores_stub() -> None:
    """#91: an installed backend is used; reset restores the stub fallback."""
    from volvence_zero.semantic_embedding import (
        reset_semantic_embedding_backend,
        semantic_embedding,
        set_semantic_embedding_backend,
        stub_semantic_embedding,
    )

    sentinel = tuple(0.5 for _ in range(8))

    class _FakeBackend:
        def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
            return tuple(0.5 for _ in range(dim))

    set_semantic_embedding_backend(_FakeBackend())
    assert semantic_embedding("anything", dim=8) == sentinel
    assert semantic_embedding("anything", dim=8) != stub_semantic_embedding(
        "anything", dim=8
    )

    reset_semantic_embedding_backend()
    assert semantic_embedding("anything", dim=8) == stub_semantic_embedding(
        "anything", dim=8
    )


def test_prototypes_are_lazy_and_reflect_injected_backend() -> None:
    """#91: prototype vectors must be embedded on demand (not frozen at
    import), so a backend injected after import is honored.
    """
    from volvence_zero.evaluation.semantic_readouts import (
        support_presence_prototype,
        task_pressure_prototype,
    )
    from volvence_zero.semantic_embedding import set_semantic_embedding_backend

    class _ConstBackend:
        def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
            return tuple(1.0 if i == 0 else 0.0 for i in range(dim))

    const = tuple(1.0 if i == 0 else 0.0 for i in range(8))
    set_semantic_embedding_backend(_ConstBackend())
    assert task_pressure_prototype() == const
    assert support_presence_prototype() == const


class _UnitBackend:
    """Backend that embeds every text onto the first basis vector."""

    def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
        return tuple(1.0 if i == 0 else 0.0 for i in range(dim))


class _OtherBackend:
    def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
        return tuple(1.0 if i == 1 else 0.0 for i in range(dim))


def test_same_owner_reinstall_is_allowed() -> None:
    """M1 (#91): the same substrate may refresh its own backend."""
    from volvence_zero.semantic_embedding import (
        semantic_embedding_backend_status,
        set_semantic_embedding_backend,
    )

    assert (
        set_semantic_embedding_backend(_UnitBackend(), owner="substrate-a")
        == "installed"
    )
    assert (
        set_semantic_embedding_backend(_UnitBackend(), owner="substrate-a")
        == "installed"
    )
    assert semantic_embedding_backend_status() == ("backend", "substrate-a", False)


def test_second_substrate_demotes_to_stub_and_latches_conflict() -> None:
    """M1 (#91): two substrates in one process must not cross-contaminate.

    A second install from a DIFFERENT owner demotes the process-global
    seam to the deterministic stub for everyone and latches an
    observable conflict flag until an explicit reset (the rollback path).
    """
    from volvence_zero.semantic_embedding import (
        reset_semantic_embedding_backend,
        semantic_embedding,
        semantic_embedding_backend_status,
        set_semantic_embedding_backend,
        stub_semantic_embedding,
    )

    set_semantic_embedding_backend(_UnitBackend(), owner="substrate-a")
    assert (
        set_semantic_embedding_backend(_OtherBackend(), owner="substrate-b")
        == "conflict-stub"
    )
    assert semantic_embedding_backend_status() == ("stub", "", True)
    assert semantic_embedding("anything", dim=8) == stub_semantic_embedding(
        "anything", dim=8
    )

    reset_semantic_embedding_backend()
    assert semantic_embedding_backend_status() == ("stub", "", False)


def test_topic_similarity_stub_fallback_matches_jaccard() -> None:
    """M1 (#91): without a backend the historical Jaccard is byte-identical."""
    from volvence_zero.semantic_embedding import (
        semantic_topic_similarity,
        stub_semantic_tokens,
    )

    pairs = (
        ("always run the failing test first", "run the failing test before edits"),
        ("完全不相关的句子", "another unrelated sentence"),
        ("", "non-empty"),
    )
    for left_text, right_text in pairs:
        left = frozenset(stub_semantic_tokens(left_text))
        right = frozenset(stub_semantic_tokens(right_text))
        if not left or not right:
            expected = 0.0
        else:
            inter = len(left & right)
            expected = inter / len(left | right) if inter else 0.0
        assert semantic_topic_similarity(left_text, right_text) == expected


def test_topic_similarity_uses_backend_cosine_when_installed() -> None:
    """M1 (#91): with a real backend, topic similarity is clamped cosine."""
    from volvence_zero.semantic_embedding import (
        semantic_topic_similarity,
        set_semantic_embedding_backend,
    )

    set_semantic_embedding_backend(_UnitBackend(), owner="substrate-a")
    # _UnitBackend maps every text to the same unit vector -> cosine 1.0,
    # even for texts with zero token overlap (proving the backend is used).
    assert semantic_topic_similarity("完全不相关的句子", "totally unrelated") == 1.0


def test_canonical_modulus_is_coprime_with_common_dims() -> None:
    """Sanity check on the SSOT modulus choice (``CANONICAL_MODULUS``).

    The hash bucket index is ``(token_index + len(token)) % dim``; if
    modulus shared a factor with dim, the value distribution would
    collapse onto a coset and bias the embedding. Guard the property
    so future modulus tweaks cannot silently regress.
    """
    from math import gcd

    from volvence_zero.semantic_embedding import CANONICAL_MODULUS

    for dim in (4, 6, 8, 16, 32, 64, 128, 256):
        assert gcd(CANONICAL_MODULUS, dim) == 1, (
            f"CANONICAL_MODULUS={CANONICAL_MODULUS} is not coprime with dim={dim}"
        )


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _defines_semantic_embedding(py_file: pathlib.Path) -> bool:
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:  # pragma: no cover
        pytest.fail(f"Cannot parse {py_file}: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_semantic_embedding":
                return True
    return False


def test_no_new_underscore_semantic_embedding_forks() -> None:
    """Reject any new ``def _semantic_embedding`` outside the allow-list.

    The three forks tracked in ``known-debts.md`` #3 are now closed:
    they import from :mod:`volvence_zero.semantic_embedding` rather than
    redefining the helper. Two unrelated helpers (memory/retrieval,
    substrate/adapter) keep their distinct signatures and are pinned in
    ``_ALLOWED_FORK_PATHS``. A new file defining ``_semantic_embedding``
    is almost certainly a regression of #3.
    """
    offenders: list[pathlib.Path] = []
    for wheel_dir in sorted(PACKAGES_ROOT.glob("vz-*")):
        for py in _python_files(wheel_dir / "src"):
            if py in _ALLOWED_FORK_PATHS:
                continue
            if _defines_semantic_embedding(py):
                offenders.append(py.relative_to(REPO_ROOT))
    if offenders:
        rendered = "\n  ".join(str(p) for p in offenders)
        pytest.fail(
            "New `def _semantic_embedding` detected outside the "
            "known-debts #3 allow-list:\n"
            f"  {rendered}\n"
            "Either reuse `volvence_zero.semantic_embedding.stub_semantic_embedding` "
            "or, if the new helper has a distinct signature warranting a "
            "separate fork, update `_ALLOWED_FORK_PATHS` together with a "
            "matching entry in `docs/known-debts.md`."
        )
