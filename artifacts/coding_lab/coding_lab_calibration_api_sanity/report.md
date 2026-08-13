# coding-lab Packet 0 calibration

- run_id: `coding_lab_calibration_api_sanity`
- hand: `api` — frozen-hand calibration
- environment_deterministic: **True**
- oracle_band: **False** (pass_rate=1.000 in [0.2, 0.8], variance_present=False)
- per-chain pass rates: [1.0]
- heldout sealed: ['heldout-0', 'heldout-1']
- episodes: 2, mean wall 73.30s, mean bytes 40777

Exit rule: any False verdict blocks Packet 1 (tune difficulty knobs first).
