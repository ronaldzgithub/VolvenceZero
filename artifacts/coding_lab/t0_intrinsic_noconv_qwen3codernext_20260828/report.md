# coding-lab Packet 0 calibration

- run_id: `t0_intrinsic_noconv_qwen3codernext_20260828`
- hand: `api` — frozen-hand calibration
- environment_deterministic: **True**
- oracle_band: **False** (pass_rate=0.931 in [0.2, 0.8], variance_present=True)
- per-chain pass rates: [1.0, 1.0, 0.9, 0.8, 1.0, 1.0, 0.7, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.9]
- heldout sealed: ['heldout-0', 'heldout-1']
- episodes: 160, mean wall 16.91s, mean bytes 45900

Exit rule: any False verdict blocks Packet 1 (tune difficulty knobs first).
