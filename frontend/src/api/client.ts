export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/$/, "");

export const PUBLIC_ASSETS_BASE_URL = (
  import.meta.env.VITE_PUBLIC_ASSETS_BASE_URL || `${API_BASE_URL}/assets`
).replace(/\/$/, "");

export class ApiError extends Error {
  status?: number;
  code?: string;
  traceId?: string;
  failedStep?: string;
  details?: unknown;

  constructor(
    message: string,
    options: {
      status?: number;
      code?: string;
      traceId?: string;
      failedStep?: string;
      details?: unknown;
    } = {},
  ) {
    super(message);
    this.name = "ApiError";
    Object.assign(this, options);
  }
}

type RequestOptions = RequestInit & { timeoutMs?: number };

export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? 30_000,
  );
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
      headers:
        options.body instanceof FormData
          ? options.headers
          : {
              "Content-Type": "application/json",
              ...options.headers,
            },
    });
    const text = await response.text();
    let payload: any = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { detail: text };
      }
    }
    if (!response.ok) {
      const error = payload?.error;
      throw new ApiError(
        error?.message || payload?.detail || `API request failed (${response.status})`,
        {
          status: response.status,
          code: error?.type,
          traceId: payload?.trace_id,
          failedStep: error?.failed_step,
          details: payload,
        },
      );
    }
    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("Request timed out.", { code: "TIMEOUT" });
    }
    throw new ApiError(
      error instanceof Error ? error.message : "Unable to reach backend.",
      { code: "NETWORK_ERROR", details: error },
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export function endpointUnavailable(error: unknown): string {
  if (error instanceof ApiError && error.status === 404) {
    return "Endpoint chưa sẵn sàng hoặc backend chưa trả dữ liệu.";
  }
  return error instanceof Error
    ? error.message
    : "Endpoint chưa sẵn sàng hoặc backend chưa trả dữ liệu.";
}
