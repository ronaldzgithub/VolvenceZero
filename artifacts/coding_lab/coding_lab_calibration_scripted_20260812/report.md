# coding-lab Packet 0 calibration

- run_id: `coding_lab_calibration_scripted_20260812`
- hand: `scripted` — machinery-only (scripted hand); frozen API hand must re-run before Packet 2 prereg
- environment_deterministic: **True**
- oracle_band: **True** (pass_rate=0.656 in [0.2, 0.8], variance_present=True)
- per-chain pass rates: [0.625, 0.875, 0.5, 0.625]
- heldout sealed: ['heldout-0', 'heldout-1']
- episodes: 32, mean wall 1.14s, mean bytes 9728

Exit rule: any False verdict blocks Packet 1 (tune difficulty knobs first).
