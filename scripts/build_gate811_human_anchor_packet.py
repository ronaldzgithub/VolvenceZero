"""Build the preregistered Gate 8/11 blinded pilot packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from volvence_zero.agent.gate811_human_anchor import (
    validate_gate811_human_anchor_preregistration,
)
from volvence_zero.agent.gate811_human_anchor_tooling import (
    build_gate811_pilot_packet,
    export_gate811_pilot_packet,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    preregistration_bytes = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_bytes)
    validate_gate811_human_anchor_preregistration(
        preregistration,
        repo_root=args.repo_root,
    )
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    bundle = build_gate811_pilot_packet(
        capture=capture,
        preregistration=preregistration,
        preregistration_sha256=hashlib.sha256(
            preregistration_bytes
        ).hexdigest(),
    )
    manifest = export_gate811_pilot_packet(
        bundle=bundle,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
