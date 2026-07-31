import { afterEach, describe, expect, it, vi } from "vitest";
import { createRun, REQUEST_TIMEOUT_MS } from "./api";
import { defaultConfig } from "./types";

function jsonResponse(payload: object, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("digital-ant API admission", () => {
  it("verifies the backend identity before creating a run", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          service: "digital-ant-app",
          schema_version: "digital-ant-app.v2",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ run_id: "run-1", status: { state: "running" } }, 201),
      );
    vi.stubGlobal("fetch", fetchMock);

    const created = await createRun(defaultConfig);

    expect(created.run_id).toBe("run-1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/health");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/runs");
  });

  it("rejects a service with an incompatible app schema", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          service: "digital-ant-app",
          schema_version: "digital-ant-app.v1",
        }),
      ),
    );

    await expect(createRun(defaultConfig)).rejects.toThrow("后端身份不匹配");
  });

  it("turns a hanging request into an actionable timeout", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((_url: string, init: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
      ),
    );

    const pending = expect(createRun(defaultConfig)).rejects.toThrow(
      "后端请求超过 10 秒",
    );
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS);
    await pending;
  });
});
