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
});
