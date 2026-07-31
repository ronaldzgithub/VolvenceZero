import type {
  AppFrame,
  DisturbanceRecord,
  ExperimentConfig,
  RunStatus,
  WorldObjectKind,
} from "./types";

export const REQUEST_TIMEOUT_MS = 10_000;
const EXPECTED_SERVICE = "digital-ant-app";
const EXPECTED_SCHEMA_VERSION = "digital-ant-app.v2";

async function requestJson<T>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT_MS,
  );
  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
    if (!response.ok) {
      throw new Error((await response.text()) || `${response.status}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(
        `后端请求超过 ${REQUEST_TIMEOUT_MS / 1000} 秒，请确认 Digital Ant 服务正在运行`,
      );
    }
    throw error;
  } finally {
    globalThis.clearTimeout(timeout);
  }
}

async function verifyBackend(): Promise<void> {
  const health = await requestJson<{
    service?: string;
    schema_version?: string;
  }>("/api/v1/health");
  if (
    health.service !== EXPECTED_SERVICE ||
    health.schema_version !== EXPECTED_SCHEMA_VERSION
  ) {
    throw new Error(
      `后端身份不匹配：需要 ${EXPECTED_SERVICE} ${EXPECTED_SCHEMA_VERSION}`,
    );
  }
}

export async function createRun(
  config: ExperimentConfig,
): Promise<{ run_id: string; status: RunStatus }> {
  await verifyBackend();
  return requestJson("/api/v1/runs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function sendCommand(
  runId: string,
  kind: "pause" | "resume" | "step" | "set_speed" | "stop",
  value?: number,
): Promise<RunStatus> {
  return requestJson(`/api/v1/runs/${runId}/commands`, {
    method: "POST",
    body: JSON.stringify({
      command_id: crypto.randomUUID(),
      kind,
      ...(value === undefined ? {} : { value }),
    }),
  });
}

export type DisturbancePayload =
  | { kind: "relocate_food"; x: number; y: number; food_index?: number }
  | { kind: "trigger_alarm"; magnitude: number; body_id?: number }
  | {
      kind: "motor_distortion";
      turn_gain: number;
      turn_bias: number;
      body_id?: number;
    }
  | {
      kind: "upsert_world_object";
      object_id: string;
      object_kind: WorldObjectKind;
      x?: number;
      y?: number;
      start_x?: number;
      start_y?: number;
      end_x?: number;
      end_y?: number;
      radius?: number;
      strength?: number;
      decay?: number;
      remaining?: number;
      angle?: number;
      length?: number;
      harm_threshold?: number;
    }
  | {
      kind: "move_world_object";
      object_id: string;
      delta_x: number;
      delta_y: number;
    }
  | { kind: "remove_world_object"; object_id: string };

export async function sendDisturbance(
  runId: string,
  disturbance: DisturbancePayload,
): Promise<DisturbanceRecord> {
  return requestJson(`/api/v1/runs/${runId}/disturbances`, {
    method: "POST",
    body: JSON.stringify({
      event_id: crypto.randomUUID(),
      ...disturbance,
    }),
  });
}

export interface StreamHandlers {
  onFrame: (sequence: number, frame: AppFrame) => void;
  onStatus: (sequence: number, status: RunStatus) => void;
  onDisturbance: (
    sequence: number,
    disturbance: DisturbanceRecord,
  ) => void;
  onConnectionError: (message: string) => void;
}

export function connectRunEvents(
  runId: string,
  after: number,
  handlers: StreamHandlers,
): EventSource {
  const source = new EventSource(
    `/api/v1/runs/${runId}/events?after=${after}`,
  );
  source.addEventListener("frame", (event) => {
    const message = event as MessageEvent<string>;
    handlers.onFrame(Number(message.lastEventId), JSON.parse(message.data));
  });
  source.addEventListener("status", (event) => {
    const message = event as MessageEvent<string>;
    handlers.onStatus(Number(message.lastEventId), JSON.parse(message.data));
  });
  source.addEventListener("disturbance", (event) => {
    const message = event as MessageEvent<string>;
    handlers.onDisturbance(
      Number(message.lastEventId),
      JSON.parse(message.data),
    );
  });
  source.onerror = () => {
    handlers.onConnectionError("事件流断开，浏览器正在自动重连");
  };
  source.onopen = () => handlers.onConnectionError("");
  return source;
}
