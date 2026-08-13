# coding-lab Packet 0 calibration

- run_id: `coding_lab_calibration_api_qwen3codernext_hard_20260813`
- hand: `api` — frozen-hand calibration
- environment_deterministic: **True**
- oracle_band: **True** (pass_rate=0.438 in [0.2, 0.8], variance_present=True)
- per-chain pass rates: [0.5, 0.625, 0.25, 0.375]
- heldout sealed: ['heldout-0', 'heldout-1']
- episodes: 32, mean wall 22.81s, mean bytes 39323

Exit rule: any False verdict blocks Packet 1 (tune difficulty knobs first).
