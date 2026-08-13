# coding-lab Packet 0 calibration

- run_id: `coding_lab_pipeline_smoke20260813_p0`
- hand: `scripted` — machinery-only (scripted hand); frozen API hand must re-run before Packet 2 prereg
- environment_deterministic: **True**
- oracle_band: **True** (pass_rate=0.750 in [0.2, 0.8], variance_present=True)
- per-chain pass rates: [0.5, 1.0]
- heldout sealed: ['heldout-0']
- episodes: 8, mean wall 2.61s, mean bytes 8869

Exit rule: any False verdict blocks Packet 1 (tune difficulty knobs first).
