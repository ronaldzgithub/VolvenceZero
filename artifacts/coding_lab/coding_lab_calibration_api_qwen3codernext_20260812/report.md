# coding-lab Packet 0 calibration

- run_id: `coding_lab_calibration_api_qwen3codernext_20260812`
- hand: `api` — frozen-hand calibration
- environment_deterministic: **True**
- oracle_band: **False** (pass_rate=0.938 in [0.2, 0.8], variance_present=True)
- per-chain pass rates: [1.0, 1.0, 0.75, 1.0]
- heldout sealed: ['heldout-0', 'heldout-1']
- episodes: 32, mean wall 174.24s, mean bytes 37517

Exit rule: any False verdict blocks Packet 1 (tune difficulty knobs first).
