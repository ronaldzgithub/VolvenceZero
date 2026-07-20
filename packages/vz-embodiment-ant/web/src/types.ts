export type AppMode = "solo" | "colony";
export type AppArm = "learned" | "no_optimize" | "fixed_rule";
export type AppObjective = "foraging" | "heading_stability";
export type AppRunState =
  | "idle"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "stopped";

export interface ExperimentConfig {
  mode: AppMode;
  arm: AppArm;
  objective: AppObjective;
  seed: number;
  n_ants: number;
  temporal_latent_dim: number;
  tick_interval_ms: number;
  max_ticks: number;
  autostart: boolean;
  food_x: number;
  food_y: number;
  motor_turn_gain: number;
  motor_turn_bias: number;
  motor_switch_tick: number | null;
  motor_switched_turn_gain: number;
  motor_switched_turn_bias: number;
}

export interface AntFrame {
  body_id: number;
  x: number;
  y: number;
  heading: number;
  target_heading: number;
  carrying_food: boolean;
  action: string;
  turn_command: number;
  applied_turn: number;
  step_command: number;
  code: number[];
  switch_gate: number;
  pe_magnitude: number;
  cumulative_credit: number;
  heading_stability_error: number;
  motor_execution_error: number;
}

export interface EvidenceProjection {
  backend_wiring: [string, string][];
  runtime_replay_captured: number;
  runtime_replay_settled: number;
  runtime_replay_transitions: number;
  runtime_replay_lineage_matches: number;
  runtime_replay_drop_reasons: string[];
  verdict: "PASS" | "BLOCK";
  verdict_reason: string;
}

export interface AppFrame {
  schema_version: string;
  run_id: string;
  sequence: number;
  tick: number;
  tick_latency_ms: number;
  mode: AppMode;
  arm: AppArm;
  objective: AppObjective;
  nest: [number, number];
  food: [number, number][];
  delivered: number;
  pickups: number;
  ants: AntFrame[];
  trail: number[][];
  evidence: EvidenceProjection;
}

export interface RunStatus {
  schema_version: string;
  run_id: string;
  state: AppRunState;
  tick: number;
  sequence: number;
  mode: AppMode;
  arm: AppArm;
  objective: AppObjective;
  seed: number;
  n_ants: number;
  tick_interval_ms: number;
  pending_disturbances: number;
  frames_retained: number;
  frames_dropped: number;
  last_error: string;
}

export interface DisturbanceRecord {
  disturbance: {
    event_id: string;
    kind: "relocate_food" | "trigger_alarm" | "motor_distortion";
  };
  status: "queued" | "applied" | "rejected";
  applied_tick: number | null;
  detail: string;
}

export interface StreamState {
  lastSequence: number;
  frame: AppFrame | null;
  status: RunStatus | null;
  disturbances: DisturbanceRecord[];
  connectionError: string;
}

export const defaultConfig: ExperimentConfig = {
  mode: "solo",
  arm: "learned",
  objective: "heading_stability",
  seed: 0,
  n_ants: 1,
  temporal_latent_dim: 16,
  tick_interval_ms: 150,
  max_ticks: 1000,
  autostart: true,
  food_x: 6,
  food_y: 0,
  motor_turn_gain: 1,
  motor_turn_bias: 0.18,
  motor_switch_tick: 30,
  motor_switched_turn_gain: 1,
  motor_switched_turn_bias: -0.18,
};
