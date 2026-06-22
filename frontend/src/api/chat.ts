import type { ChatResponse } from "../types";
import { apiRequest } from "./client";

export function sendChat(message: string, sessionId: string): Promise<ChatResponse> {
  return apiRequest<ChatResponse>("/chat", {
    method: "POST",
    // Local demo models may take several minutes. Let the request finish and
    // rely on backend/model-provider timeouts instead.
    timeoutMs: null,
    body: JSON.stringify({
      message,
      session_id: sessionId,
      history: [],
      debug: true,
    }),
  });
}
