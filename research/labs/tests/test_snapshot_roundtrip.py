"""CAS put/get roundtrip + idempotence + snapshot immutability."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from volvence_labs.framework.snapshot import CASStore, canonical_dumps, default_paths


class TestSnapshotRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = default_paths(self.tmp.name)
        self.store = CASStore(self.paths)
        self.addCleanup(self.store.close)

    def test_bytes_roundtrip_and_idempotent(self) -> None:
        sha1 = self.store.put_bytes(b"hello world", kind="raw")
        sha2 = self.store.put_bytes(b"hello world", kind="raw")
        self.assertEqual(sha1, sha2)
        self.assertEqual(self.store.get_bytes(sha1), b"hello world")
        self.assertTrue(self.store.exists(sha1))

    def test_obj_canonical_json_stable_across_key_order(self) -> None:
        a = {"x": 1, "y": [2, 3], "z": "abc"}
        b = {"z": "abc", "y": [2, 3], "x": 1}
        self.assertEqual(canonical_dumps(a), canonical_dumps(b))
        sha_a = self.store.put_obj(a, kind="test")
        sha_b = self.store.put_obj(b, kind="test")
        self.assertEqual(sha_a, sha_b)
        self.assertEqual(self.store.get_obj(sha_a), a)

    def test_distinct_objects_distinct_shas(self) -> None:
        sha1 = self.store.put_obj({"a": 1}, kind="test")
        sha2 = self.store.put_obj({"a": 2}, kind="test")
        self.assertNotEqual(sha1, sha2)

    def test_immutable_on_disk(self) -> None:
        sha = self.store.put_obj({"a": 1}, kind="test")
        path = self.paths.cas_dir / sha[:2] / f"{sha}.bin"
        self.assertTrue(path.exists())
        original = path.read_bytes()
        # Overwriting via put_* with same content must be a no-op byte-wise.
        self.store.put_obj({"a": 1}, kind="test")
        self.assertEqual(path.read_bytes(), original)

    def test_nan_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.put_obj({"x": float("nan")}, kind="test")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
