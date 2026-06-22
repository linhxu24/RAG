import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiRequest } from "./client";

afterEach(() => vi.restoreAllMocks());

describe("apiRequest", () => {
  it("normalizes backend errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Endpoint unavailable" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await expect(apiRequest("/missing")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "Endpoint unavailable",
    } satisfies Partial<ApiError>);
  });

  it("supports requests without a client-side timeout", async () => {
    const timeoutSpy = vi.spyOn(window, "setTimeout");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(
      apiRequest<{ ok: boolean }>("/slow", { timeoutMs: null }),
    ).resolves.toEqual({ ok: true });
    expect(timeoutSpy).not.toHaveBeenCalled();
  });
});
