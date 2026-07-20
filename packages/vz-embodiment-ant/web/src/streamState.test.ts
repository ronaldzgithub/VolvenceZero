import { describe, expect, it } from "vitest";
import {
  initialStreamState,
  reduceStreamState,
} from "./streamState";
import type { AppFrame, RunStatus } from "./types";

const status: RunStatus = {
  schema_version: "digital-ant-app.v2",
  run_id: "r1",
  state: "running",
  tick: 0,
  sequence: 1,
  mode: "solo",
  arm: "learned",
  objective: "heading_stability",
  seed: 0,
  n_ants: 1,
  tick_interval_ms: 100,
  pending_disturbances: 0,
  frames_retained: 0,
  frames_dropped: 0,
  last_error: "",
};

const frame: AppFrame = {
  schema_version: "digital-ant-app.v2",
  run_id: "r1",
  sequence: 2,
  tick: 1,
  tick_latency_ms: 12.5,
  mode: "solo",
  arm: "learned",
  objective: "heading_stability",
  nest: [0, 0],
  food: [[6, 0]],
  delivered: 0,
  pickups: 0,
  ants: [],
  trail: [],
  objects: [],
  evidence: {
    backend_wiring: [],
    runtime_replay_captured: 0,
    runtime_replay_settled: 0,
    runtime_replay_transitions: 0,
    runtime_replay_lineage_matches: 0,
    runtime_replay_drop_reasons: [],
    verdict: "BLOCK",
    verdict_reason: "not passed",
    checkpoint_loaded: false,
    checkpoint_fingerprint: "",
    checkpoint_verdict: "UNAVAILABLE",
  },
};

describe("SSE reducer", () => {
  it("deduplicates reconnect replay by monotonic sequence", () => {
    const withStatus = reduceStreamState(initialStreamState, {
      type: "status",
      sequence: 1,
      payload: status,
    });
    const withFrame = reduceStreamState(withStatus, {
      type: "frame",
      sequence: 2,
      payload: frame,
    });
    const replayedOldStatus = reduceStreamState(withFrame, {
      type: "status",
      sequence: 1,
      payload: { ...status, state: "failed" },
    });

    expect(replayedOldStatus).toBe(withFrame);
    expect(replayedOldStatus.frame?.tick).toBe(1);
    expect(replayedOldStatus.status?.state).toBe("running");
  });

  it("keeps only the latest twenty disturbance records", () => {
    let state = initialStreamState;
    for (let sequence = 1; sequence <= 25; sequence += 1) {
      state = reduceStreamState(state, {
        type: "disturbance",
        sequence,
        payload: {
          disturbance: {
            event_id: `event-${sequence}`,
            kind: "trigger_alarm",
          },
          status: "applied",
          applied_tick: sequence,
          detail: "ok",
        },
      });
    }
    expect(state.disturbances).toHaveLength(20);
    expect(state.disturbances[0].disturbance.event_id).toBe("event-6");
  });
});
