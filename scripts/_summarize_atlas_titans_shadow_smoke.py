"""Summarize the existing CMS ATLAS / Titans SHADOW smoke JSON.

Reads ``artifacts/cms_atlas_titans_shadow_smoke/atlas_titans_cms_shadow_smoke.json``
and prints (a) aggregate metric deltas vs the ``pe-eta`` baseline, (b)
per-case CMS-sensitive deltas, (c) the canonical ``passed`` flag for each
profile.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    src = Path("artifacts/cms_atlas_titans_shadow_smoke/atlas_titans_cms_shadow_smoke.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    canonical = data["per_path_metric_means"]["pe-eta"]
    uplift = data["per_path_metric_means"]["atlas-titans-cms-uplift"]

    diffs = []
    for k in sorted(canonical):
        if k not in uplift:
            continue
        d = uplift[k] - canonical[k]
        if abs(d) > 1e-6:
            diffs.append((k, canonical[k], uplift[k], d))

    print(f"Aggregate diverged metrics: {len(diffs)} / {len(canonical)}")
    for k, c, u, d in diffs:
        sign = "+" if d >= 0 else ""
        print(f"  {k:50s}  canonical={c:.4f}  uplift={u:.4f}  delta={sign}{d:.4f}")

    print()
    print("Per-case CMS-sensitive deltas (uplift vs pe-eta):")
    for case_entry in data["case_deltas_from_baseline"]:
        case_id = case_entry["case_id"]
        uplift_deltas = case_entry["paths"].get("atlas-titans-cms-uplift", {})
        nontrivial = {k: v for k, v in uplift_deltas.items() if abs(v) > 1e-6}
        if not nontrivial:
            continue
        print(f"  case={case_id}")
        for k in sorted(nontrivial):
            sign = "+" if nontrivial[k] >= 0 else ""
            print(f"    {k}: {sign}{nontrivial[k]:.4f}")

    print()
    print("Canonical acceptance verdict per profile:")
    for path in data["per_path_metric_means"]:
        passed = data["per_path_metric_means"][path].get("passed")
        print(f"  {path}: passed={passed}")


if __name__ == "__main__":
    main()
