import { useEffect, useReducer, useRef, useState } from "react";
import {
  connectRunEvents,
  createRun,
  sendCommand,
  sendDisturbance,
} from "./api";
import {
  initialStreamState,
  reduceStreamState,
} from "./streamState";
import {
  defaultConfig,
  type AppArm,
  type AppMode,
  type AppObjective,
  type ExperimentConfig,
} from "./types";
import { WorldCanvas } from "./WorldCanvas";

function numberValue(value: string): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export default function App() {
  const [config, setConfig] = useState<ExperimentConfig>(defaultConfig);
  const [runId, setRunId] = useState("");
  const [stream, dispatch] = useReducer(
    reduceStreamState,
    initialStreamState,
  );
  const [requestError, setRequestError] = useState("");
  const [busy, setBusy] = useState(false);
  const [foodX, setFoodX] = useState(-4);
  const [foodY, setFoodY] = useState(4);
  const [alarm, setAlarm] = useState(1);
  const [motorGain, setMotorGain] = useState(1);
  const [motorBias, setMotorBias] = useState(-0.18);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!runId) return;
    sourceRef.current?.close();
    const source = connectRunEvents(runId, 0, {
      onFrame: (sequence, payload) =>
        dispatch({ type: "frame", sequence, payload }),
      onStatus: (sequence, payload) =>
        dispatch({ type: "status", sequence, payload }),
      onDisturbance: (sequence, payload) =>
        dispatch({ type: "disturbance", sequence, payload }),
      onConnectionError: (message) =>
        dispatch({ type: "connection", message }),
    });
    sourceRef.current = source;
    return () => source.close();
  }, [runId]);

  async function withRequest(action: () => Promise<void>) {
    setRequestError("");
    try {
      await action();
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : String(error));
    }
  }

  async function startRun() {
    setBusy(true);
    await withRequest(async () => {
      sourceRef.current?.close();
      dispatch({ type: "reset" });
      const created = await createRun(config);
      setRunId(created.run_id);
      dispatch({
        type: "status",
        sequence: created.status.sequence,
        payload: created.status,
      });
    });
    setBusy(false);
  }

  async function command(
    kind: "pause" | "resume" | "step" | "stop",
  ) {
    if (!runId) return;
    await withRequest(async () => {
      const status = await sendCommand(runId, kind);
      dispatch({
        type: "status",
        sequence: status.sequence,
        payload: status,
      });
    });
  }

  async function updateSpeed() {
    if (!runId) return;
    await withRequest(async () => {
      const nextStatus = await sendCommand(
        runId,
        "set_speed",
        config.tick_interval_ms,
      );
      dispatch({
        type: "status",
        sequence: nextStatus.sequence,
        payload: nextStatus,
      });
    });
  }

  function setMode(mode: AppMode) {
    setConfig((current) => ({
      ...current,
      mode,
      n_ants: mode === "solo" ? 1 : Math.max(3, current.n_ants),
      objective:
        mode === "colony" ? "foraging" : current.objective,
    }));
  }

  const frame = stream.frame;
  const status = stream.status;
  const evidence = frame?.evidence;
  const verdict = evidence?.verdict ?? "BLOCK";
  const activeAnt = frame?.ants[0];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">VOLVENCE · NON-LANGUAGE EMBODIMENT</p>
          <h1>数字蚂蚁实时实验场</h1>
        </div>
        <div className="status-stack">
          <span className={`live-dot ${status?.state ?? "idle"}`} />
          <strong>{status?.state ?? "未启动"}</strong>
          <span>tick {frame?.tick ?? 0}</span>
        </div>
      </header>

      <main className="layout">
        <aside className="control-panel">
          <section>
            <h2>实验形态</h2>
            <div className="segmented">
              {(["solo", "colony"] as AppMode[]).map((mode) => (
                <button
                  className={config.mode === mode ? "selected" : ""}
                  key={mode}
                  onClick={() => setMode(mode)}
                >
                  {mode === "solo" ? "单蚁校准" : "群体生态"}
                </button>
              ))}
            </div>
            <label>
              控制臂
              <select
                value={config.arm}
                onChange={(event) =>
                  setConfig({
                    ...config,
                    arm: event.target.value as AppArm,
                  })
                }
              >
                <option value="learned">learned kernel</option>
                <option value="no_optimize">no-optimize</option>
                <option value="fixed_rule">fixed-rule baseline</option>
              </select>
            </label>
            <label>
              目标
              <select
                value={config.objective}
                disabled={config.mode === "colony"}
                onChange={(event) =>
                  setConfig({
                    ...config,
                    objective: event.target.value as AppObjective,
                  })
                }
              >
                <option value="heading_stability">航向稳定</option>
                <option value="foraging">觅食</option>
              </select>
            </label>
            <div className="form-grid">
              <label>
                seed
                <input
                  type="number"
                  value={config.seed}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      seed: numberValue(event.target.value),
                    })
                  }
                />
              </label>
              <label>
                蚂蚁数
                <input
                  type="number"
                  min={1}
                  max={16}
                  disabled={config.mode === "solo"}
                  value={config.n_ants}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      n_ants: Math.max(1, numberValue(event.target.value)),
                    })
                  }
                />
              </label>
              <label>
                tick 间隔 ms
                <input
                  type="number"
                  min={0}
                  value={config.tick_interval_ms}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      tick_interval_ms: Math.max(
                        0,
                        numberValue(event.target.value),
                      ),
                    })
                  }
                />
              </label>
              <label>
                最大 tick
                <input
                  type="number"
                  min={1}
                  value={config.max_ticks}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      max_ticks: Math.max(1, numberValue(event.target.value)),
                    })
                  }
                />
              </label>
            </div>
            <button
              className="primary-action"
              onClick={startRun}
              disabled={busy}
            >
              {busy ? "创建中…" : "创建真实闭环"}
            </button>
            <div className="transport">
              <button onClick={() => command("pause")} disabled={!runId}>
                暂停
              </button>
              <button onClick={() => command("step")} disabled={!runId}>
                单步
              </button>
              <button onClick={() => command("resume")} disabled={!runId}>
                继续
              </button>
              <button onClick={updateSpeed} disabled={!runId}>
                应用速度
              </button>
              <button onClick={() => command("stop")} disabled={!runId}>
                停止
              </button>
            </div>
          </section>

          <section>
            <h2>环境扰动</h2>
            <p className="section-note">
              只在 tick / round 边界进入环境 owner，不直接写控制器。
            </p>
            <div className="inline-controls">
              <input
                aria-label="食物 X"
                type="number"
                value={foodX}
                onChange={(event) => setFoodX(numberValue(event.target.value))}
              />
              <input
                aria-label="食物 Y"
                type="number"
                value={foodY}
                onChange={(event) => setFoodY(numberValue(event.target.value))}
              />
              <button
                disabled={!runId}
                onClick={() =>
                  withRequest(async () => {
                    await sendDisturbance(runId, {
                      kind: "relocate_food",
                      x: foodX,
                      y: foodY,
                    });
                  })
                }
              >
                搬迁食物
              </button>
            </div>
            <div className="inline-controls two">
              <input
                aria-label="报警强度"
                type="number"
                step="0.1"
                value={alarm}
                onChange={(event) => setAlarm(numberValue(event.target.value))}
              />
              <button
                disabled={!runId}
                onClick={() =>
                  withRequest(async () => {
                    await sendDisturbance(runId, {
                      kind: "trigger_alarm",
                      magnitude: alarm,
                    });
                  })
                }
              >
                触发 alarm
              </button>
            </div>
            <div className="inline-controls">
              <input
                aria-label="电机增益"
                type="number"
                step="0.05"
                value={motorGain}
                onChange={(event) =>
                  setMotorGain(numberValue(event.target.value))
                }
              />
              <input
                aria-label="电机偏置"
                type="number"
                step="0.01"
                value={motorBias}
                onChange={(event) =>
                  setMotorBias(numberValue(event.target.value))
                }
              />
              <button
                disabled={!runId}
                onClick={() =>
                  withRequest(async () => {
                    await sendDisturbance(runId, {
                      kind: "motor_distortion",
                      turn_gain: motorGain,
                      turn_bias: motorBias,
                    });
                  })
                }
              >
                施加隐藏电机扰动
              </button>
            </div>
          </section>

          <section className="truth-boundary">
            <h2>证据边界</h2>
            <div className={`verdict ${verdict.toLowerCase()}`}>
              {verdict}
            </div>
            <p>
              {evidence?.verdict_reason ??
                "尚无通过冻结门槛的正式 artifact；画面仍是真实内核行动。"}
            </p>
            {runId && (
              <a
                href={`/api/v1/runs/${runId}/replay`}
                target="_blank"
                rel="noreferrer"
              >
                导出不可变 replay
              </a>
            )}
          </section>
        </aside>

        <section className="stage">
          <WorldCanvas frame={frame} />
          <div className="metrics-strip">
            <article>
              <span>pickup / delivery</span>
              <strong>
                {frame?.pickups ?? 0} / {frame?.delivered ?? 0}
              </strong>
            </article>
            <article>
              <span>动作</span>
              <strong>{activeAnt?.action ?? "—"}</strong>
            </article>
            <article>
              <span>β switch</span>
              <strong>{activeAnt?.switch_gate.toFixed(3) ?? "—"}</strong>
            </article>
            <article>
              <span>PE / credit</span>
              <strong>
                {activeAnt
                  ? `${activeAnt.pe_magnitude.toFixed(3)} / ${activeAnt.cumulative_credit.toFixed(3)}`
                  : "—"}
              </strong>
            </article>
            <article>
              <span>replay settled</span>
              <strong>{evidence?.runtime_replay_settled ?? 0}</strong>
            </article>
            <article>
              <span>实际 tick latency</span>
              <strong>
                {frame ? `${frame.tick_latency_ms.toFixed(1)} ms` : "—"}
              </strong>
            </article>
          </div>

          <div className="lower-grid">
            <section className="telemetry">
              <h2>真实动作传导</h2>
              <dl>
                <div>
                  <dt>z_t</dt>
                  <dd>
                    {activeAnt?.code.length
                      ? `[${activeAnt.code
                          .slice(0, 8)
                          .map((value) => value.toFixed(2))
                          .join(", ")}${activeAnt.code.length > 8 ? ", …" : ""}]`
                      : "baseline / unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>commanded / applied / step</dt>
                  <dd>
                    {activeAnt
                      ? `${activeAnt.turn_command.toFixed(3)} / ${activeAnt.applied_turn.toFixed(3)} / ${activeAnt.step_command.toFixed(3)}`
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>heading error</dt>
                  <dd>
                    {activeAnt?.heading_stability_error.toFixed(4) ?? "—"}
                  </dd>
                </div>
                <div>
                  <dt>heading / target</dt>
                  <dd>
                    {activeAnt
                      ? `${activeAnt.heading.toFixed(3)} / ${activeAnt.target_heading.toFixed(3)}`
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>motor execution error</dt>
                  <dd>
                    {activeAnt?.motor_execution_error.toFixed(4) ?? "—"}
                  </dd>
                </div>
                <div>
                  <dt>runtime replay lineage</dt>
                  <dd>
                    {evidence
                      ? `${evidence.runtime_replay_lineage_matches}/${evidence.runtime_replay_transitions}`
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>backend wiring</dt>
                  <dd>
                    {evidence?.backend_wiring.length
                      ? evidence.backend_wiring
                          .map(([name, value]) => `${name}=${value}`)
                          .join(" · ")
                      : "baseline / unavailable"}
                  </dd>
                </div>
              </dl>
            </section>
            <section className="event-log">
              <h2>扰动审计</h2>
              {stream.disturbances.length ? (
                <ol>
                  {stream.disturbances
                    .slice()
                    .reverse()
                    .map((event, index) => (
                      <li key={`${event.disturbance.event_id}-${index}`}>
                        <strong>{event.disturbance.kind}</strong>
                        <span>
                          {event.status}
                          {event.applied_tick === null
                            ? ""
                            : ` @ tick ${event.applied_tick}`}
                        </span>
                      </li>
                    ))}
                </ol>
              ) : (
                <p>尚未注入扰动</p>
              )}
            </section>
          </div>
        </section>
      </main>

      {(requestError || stream.connectionError || status?.last_error) && (
        <div className="error-toast" role="alert">
          {requestError || status?.last_error || stream.connectionError}
        </div>
      )}
    </div>
  );
}
