import type { ChatResponse } from "../types";
import { apiRequest } from "./client";

export function sendChat(message: string, sessionId: string): Promise<ChatResponse> {
  return apiRequest<ChatResponse>("/chat", {
    method: "POST",
    timeoutMs: 130_000,
    body: JSON.stringify({
      message,
      session_id: sessionId,
      history: [],
      debug: true,
    }),
  });
}
