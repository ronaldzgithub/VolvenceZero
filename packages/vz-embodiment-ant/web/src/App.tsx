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
  type AppFrame,
  type AppMode,
  type AppObjective,
  type AppRunState,
  type EvidenceProjection,
  type ExperimentConfig,
  type RunStatus,
  type WorldObjectKind,
  type WorldObjectSnapshot,
} from "./types";
import { type CanvasTool, WorldCanvas } from "./WorldCanvas";

function numberValue(value: string): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function isTerminalRunState(state: AppRunState | undefined): boolean {
  return state === "completed" || state === "failed" || state === "stopped";
}

function statusLabel(state: AppRunState | undefined): string {
  if (state === "running") return "观察中";
  if (state === "paused") return "已暂停";
  if (state === "completed") return "已完成";
  if (state === "failed") return "运行异常";
  if (state === "stopped") return "已停止";
  return "未启动";
}

function stageTitle(frame: AppFrame | null, status: RunStatus | null): string {
  if (!frame) return "等待蚁群进入环境";
  const colony = `${frame.ants.length} 只蚂蚁`;
  if (status?.state === "paused") return `${colony}已暂停，保留当前环境帧`;
  if (status?.state === "stopped") return `${colony}已停止，保留最后环境帧`;
  if (status?.state === "completed") return `${colony}已完成本次实验`;
  if (status?.state === "failed") return `${colony}的实验运行异常`;
  if (frame.arm === "fixed_rule") return `${colony}正在按固定规则行动`;
  if (frame.arm === "no_optimize") return `${colony}正在运行无优化对照`;
  if (frame.evidence.checkpoint_loaded) return `${colony}正在使用晋级 checkpoint 行动`;
  return `${colony}正在从冷启动在线学习`;
}

function checkpointLabel(
  arm: AppArm,
  evidence: EvidenceProjection | undefined,
): string {
  if (arm === "fixed_rule") return "不适用（固定规则基线）";
  if (arm === "no_optimize") return "不加载（no-optimize 对照）";
  return evidence?.checkpoint_loaded
    ? `${evidence.checkpoint_verdict} · ${evidence.checkpoint_fingerprint.slice(0, 12)}`
    : "未加载（冷启动在线学习）";
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
  const [selectedAntId, setSelectedAntId] = useState(0);
  const sourceRef = useRef<EventSource | null>(null);

  const frame = stream.frame;
  const status = stream.status;
  const evidence = frame?.evidence;
  const verdict = evidence?.verdict ?? "BLOCK";
  const runIsTerminal = isTerminalRunState(status?.state);
  const hasActiveRun = Boolean(runId && status && !runIsTerminal);
  const configLocked = busy || hasActiveRun;
  const canEditEcology = Boolean(
    runId && status && !runIsTerminal && status.objective === "ecology",
  );
  const activeAnt =
    frame?.ants.find((ant) => ant.body_id === selectedAntId) ?? frame?.ants[0];
  const activeArm = frame?.arm ?? status?.arm ?? config.arm;

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
      if (runId && status && !isTerminalRunState(status.state)) {
        await sendCommand(runId, "stop");
      }
      sourceRef.current?.close();
      setRunId("");
      dispatch({ type: "reset" });
      setSelectedObjectId(null);
      setSelectedAntId(0);
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
    if (!canEditEcology) return;
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
    if (!canEditEcology) return;
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
    if (!canEditEcology || !selectedObjectId) return;
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
            {statusLabel(status?.state)}
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
                  disabled={configLocked}
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
                disabled={configLocked}
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
                disabled={configLocked || config.mode === "colony"}
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
                  disabled={configLocked}
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
                  disabled={configLocked || config.mode === "solo"}
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
                  disabled={busy}
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
                  disabled={configLocked}
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
              disabled={configLocked}
            >
              {busy
                ? "正在唤醒蚁群…"
                : hasActiveRun
                  ? "当前闭环运行中"
                  : "创建真实闭环"}
            </button>
            <div className="transport">
              <button onClick={() => command("pause")} disabled={status?.state !== "running"}>
                暂停
              </button>
              <button onClick={() => command("step")} disabled={status?.state !== "paused"}>
                单步
              </button>
              <button onClick={() => command("resume")} disabled={status?.state !== "paused"}>
                继续
              </button>
              <button onClick={updateSpeed} disabled={!hasActiveRun}>
                应用速度
              </button>
              <button onClick={() => command("stop")} disabled={!hasActiveRun}>
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
                disabled={!hasActiveRun || status?.objective === "ecology"}
                value={foodX}
                onChange={(event) => setFoodX(numberValue(event.target.value))}
              />
              <input
                aria-label="食物 Y"
                type="number"
                disabled={!hasActiveRun || status?.objective === "ecology"}
                value={foodY}
                onChange={(event) => setFoodY(numberValue(event.target.value))}
              />
              <button
                disabled={!hasActiveRun || status?.objective === "ecology"}
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
                disabled={!hasActiveRun}
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
                disabled={!hasActiveRun}
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
              checkpoint：{checkpointLabel(activeArm, evidence)}
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
              <h2>{stageTitle(frame, status)}</h2>
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
                  disabled={!canEditEcology}
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
              disabled={!canEditEcology || !selectedObjectId}
            >
              移除选中物体
            </button>
          </div>

          <div className="canvas-shell">
            <WorldCanvas
              frame={frame}
              interactionEnabled={canEditEcology}
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
              <span>群体拾取 / 回巢</span>
              <strong>
                {frame?.pickups ?? 0} / {frame?.delivered ?? 0}
              </strong>
            </article>
            <article>
              <span>蚂蚁 #{activeAnt?.body_id ?? "—"} 动作</span>
              <strong>{activeAnt?.action ?? "—"}</strong>
            </article>
            <article>
              <span>蚂蚁 #{activeAnt?.body_id ?? "—"} 策略切换 β</span>
              <strong>{activeAnt?.switch_gate.toFixed(3) ?? "—"}</strong>
            </article>
            <article>
              <span>蚂蚁 #{activeAnt?.body_id ?? "—"} PE / 信用</span>
              <strong>
                {activeAnt
                  ? `${activeAnt.pe_magnitude.toFixed(3)} / ${activeAnt.cumulative_credit.toFixed(3)}`
                  : "—"}
              </strong>
            </article>
            <article>
              <span>群体闭环结算 / 可用</span>
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
              <span>群体 round 耗时</span>
              <strong>
                {frame ? `${frame.tick_latency_ms.toFixed(1)} ms` : "—"}
              </strong>
            </article>
          </div>

          <div className="lower-grid">
            <section className="telemetry">
              <div className="telemetry-heading">
                <h2>单蚁真实动作传导</h2>
                <label>
                  遥测对象
                  <select
                    aria-label="遥测对象"
                    value={activeAnt?.body_id ?? 0}
                    disabled={!frame || frame.ants.length <= 1}
                    onChange={(event) =>
                      setSelectedAntId(numberValue(event.target.value))
                    }
                  >
                    {(frame?.ants ?? []).map((ant) => (
                      <option key={ant.body_id} value={ant.body_id}>
                        蚂蚁 #{ant.body_id}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
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
