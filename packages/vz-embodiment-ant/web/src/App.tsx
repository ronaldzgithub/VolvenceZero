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
  type WorldObjectKind,
  type WorldObjectSnapshot,
} from "./types";
import { type CanvasTool, WorldCanvas } from "./WorldCanvas";

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
  const [canvasTool, setCanvasTool] = useState<CanvasTool>("butter");
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(
    null,
  );
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
      setSelectedObjectId(null);
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

  function placeObject(
    kind: WorldObjectKind,
    start: [number, number],
    end?: [number, number],
  ) {
    if (!runId) return;
    const objectId = `${kind}-${crypto.randomUUID()}`;
    void withRequest(async () => {
      if (kind === "wood_stick") {
        if (!end) throw new Error("木棍需要拖出方向和长度");
        await sendDisturbance(runId, {
          kind: "upsert_world_object",
          object_id: objectId,
          object_kind: kind,
          start_x: start[0],
          start_y: start[1],
          end_x: end[0],
          end_y: end[1],
          radius: 0.22,
        });
      } else if (kind === "butter") {
        await sendDisturbance(runId, {
          kind: "upsert_world_object",
          object_id: objectId,
          object_kind: kind,
          x: start[0],
          y: start[1],
          strength: 1.6,
          decay: 4,
          radius: 1.2,
        });
      } else {
        await sendDisturbance(runId, {
          kind: "upsert_world_object",
          object_id: objectId,
          object_kind: kind,
          x: start[0],
          y: start[1],
          strength: 1,
          decay: 1.8,
          harm_threshold: 0.55,
        });
      }
      setSelectedObjectId(objectId);
    });
  }

  function moveObject(
    object: WorldObjectSnapshot,
    delta: [number, number],
  ) {
    if (!runId) return;
    void withRequest(async () => {
      await sendDisturbance(runId, {
        kind: "move_world_object",
        object_id: object.object_id,
        delta_x: delta[0],
        delta_y: delta[1],
      });
    });
  }

  function removeSelectedObject() {
    if (!runId || !selectedObjectId) return;
    void withRequest(async () => {
      await sendDisturbance(runId, {
        kind: "remove_world_object",
        object_id: selectedObjectId,
      });
      setSelectedObjectId(null);
    });
  }

  function setMode(mode: AppMode) {
    setConfig((current) => ({
      ...current,
      mode,
      n_ants: mode === "solo" ? 1 : Math.max(3, current.n_ants),
      objective:
        mode === "colony" ? "ecology" : current.objective,
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
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">V</span>
          <div>
            <p className="eyebrow">VOLVENCE · DIGITAL COLONY</p>
            <h1>数字蚁巢观察站</h1>
          </div>
        </div>
        <div className="status-stack">
          <span className={`live-dot ${status?.state ?? "idle"}`} />
          <strong>
            {status?.state === "running"
              ? "观察中"
              : status?.state === "paused"
                ? "已暂停"
                : status?.state === "completed"
                  ? "已完成"
                  : status?.state === "failed"
                    ? "运行异常"
                    : "未启动"}
          </strong>
          <span>tick {frame?.tick ?? 0}</span>
        </div>
      </header>

      <main className="layout">
        <aside className="control-panel">
          <section className="setup-panel">
            <p className="panel-kicker">观察设置</p>
            <h2>创建一个真实蚁群</h2>
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
                <option value="ecology">三物体生态</option>
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
              {busy ? "正在唤醒蚁群…" : "创建真实闭环"}
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

          <details className="advanced-panel">
            <summary>高级环境扰动</summary>
            <section>
            <h2>实验扰动</h2>
            <p className="section-note">
              只在 tick / round 边界进入环境 owner，不直接写控制器。
            </p>
            <div className="inline-controls">
              <input
                aria-label="食物 X"
                type="number"
                disabled={config.objective === "ecology"}
                value={foodX}
                onChange={(event) => setFoodX(numberValue(event.target.value))}
              />
              <input
                aria-label="食物 Y"
                type="number"
                disabled={config.objective === "ecology"}
                value={foodY}
                onChange={(event) => setFoodY(numberValue(event.target.value))}
              />
              <button
                disabled={!runId || config.objective === "ecology"}
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
          </details>

          <section className="truth-boundary">
            <div className="truth-title">
              <h2>行为证据</h2>
              <div className={`verdict ${verdict.toLowerCase()}`}>
                {verdict}
              </div>
            </div>
            <p>
              {evidence?.verdict_reason ??
                "尚无通过冻结门槛的正式 artifact；画面仍是真实内核行动。"}
            </p>
            <p className="checkpoint-status">
              learned checkpoint：
              {evidence?.checkpoint_loaded
                ? `${evidence.checkpoint_verdict} · ${evidence.checkpoint_fingerprint.slice(0, 12)}`
                : "未加载（冷启动）"}
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
          <div className="stage-heading">
            <div>
              <p className="panel-kicker">实时生态观察箱</p>
              <h2>
                {frame ? `${frame.ants.length} 只蚂蚁正在自主行动` : "等待蚁群进入环境"}
              </h2>
            </div>
            <div className="field-readout">
              <span>拾取 <strong>{frame?.pickups ?? 0}</strong></span>
              <span>送回巢穴 <strong>{frame?.delivered ?? 0}</strong></span>
            </div>
          </div>

          <div className="field-toolbar" aria-label="生态物体工具">
            <div className="object-tools">
              {(
                [
                  ["butter", "黄油", "butter"],
                  ["wood_stick", "木棍", "stick"],
                  ["burning_match", "燃烧火柴", "match"],
                  ["select", "选择 / 移动", "cursor"],
                ] as [CanvasTool, string, string][]
              ).map(([tool, label, icon]) => (
                <button
                  key={tool}
                  className={canvasTool === tool ? "selected" : ""}
                  onClick={() => setCanvasTool(tool)}
                  disabled={!runId || config.objective !== "ecology"}
                  aria-pressed={canvasTool === tool}
                >
                  <span className={`tool-icon ${icon}`} aria-hidden="true" />
                  <span>{label}</span>
                </button>
              ))}
            </div>
            <p className="tool-hint">
              {canvasTool === "wood_stick"
                ? "在环境中按住并拖动，画出木棍"
                : canvasTool === "select"
                  ? "拖动物体改变环境，点击空地取消选择"
                  : `点击环境，放下一块${canvasTool === "butter" ? "黄油" : "燃烧火柴"}`}
            </p>
            <button
              className="delete-object"
              onClick={removeSelectedObject}
              disabled={!runId || !selectedObjectId}
            >
              移除选中物体
            </button>
          </div>

          <div className="canvas-shell">
            <WorldCanvas
              frame={frame}
              tool={canvasTool}
              selectedObjectId={selectedObjectId}
              onPlaceObject={placeObject}
              onMoveObject={moveObject}
              onSelectObject={setSelectedObjectId}
            />
            <div className="canvas-legend" aria-hidden="true">
              <span><i className="legend-dot pheromone" />信息素路径</span>
              <span><i className="legend-dot heat" />有害热区</span>
            </div>
          </div>
          <div className="metrics-strip">
            <article>
              <span>拾取 / 回巢</span>
              <strong>
                {frame?.pickups ?? 0} / {frame?.delivered ?? 0}
              </strong>
            </article>
            <article>
              <span>动作</span>
              <strong>{activeAnt?.action ?? "—"}</strong>
            </article>
            <article>
              <span>策略切换 β</span>
              <strong>{activeAnt?.switch_gate.toFixed(3) ?? "—"}</strong>
            </article>
            <article>
              <span>预测误差 / 信用</span>
              <strong>
                {activeAnt
                  ? `${activeAnt.pe_magnitude.toFixed(3)} / ${activeAnt.cumulative_credit.toFixed(3)}`
                  : "—"}
              </strong>
            </article>
            <article>
              <span>闭环结算 / 可用</span>
              <strong>
                {evidence
                  ? `${evidence.runtime_replay_settled}/${Math.max(
                      0,
                      evidence.runtime_replay_captured -
                        evidence.runtime_replay_pending_captures,
                    )}`
                  : "0/0"}
              </strong>
            </article>
            <article>
              <span>每步耗时</span>
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
