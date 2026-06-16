import { AlertTriangle, Bot, ThumbsDown, ThumbsUp } from "lucide-react";

import type { ChatMessage } from "../../types";
import { AssetGallery } from "./AssetGallery";
import { SourceCards } from "./SourceCards";

export function AssistantMessageCard({ message }: { message: ChatMessage }) {
  const response = message.response;
  const answer = response?.answer || response?.message;
  const error = response?.error;
  return (
    <div className="mx-auto w-full max-w-[760px]">
      <div
        className={`rounded-2xl border p-5 shadow-sm ${
          message.error || error
            ? "border-rose-200 bg-rose-50/70"
            : "border-slate-200 bg-white"
        }`}
      >
        <div className="mb-3 flex items-center gap-2">
          <span
            className={`grid h-8 w-8 place-items-center rounded-xl ${
              message.error || error
                ? "bg-rose-100 text-rose-600"
                : "bg-teal-50 text-teal-700"
            }`}
          >
            {message.error || error ? <AlertTriangle size={17} /> : <Bot size={17} />}
          </span>
          <div>
            <div className="text-xs font-bold text-slate-800">
              {message.error || error ? "Chatbot request failed" : "SimplyDent Assistant"}
            </div>
            <div className="text-[10px] text-slate-400">
              {response?.intent || "Dental support"}
            </div>
          </div>
        </div>
        <div className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
          {message.error || error?.message || answer?.text || message.text}
        </div>
        {answer?.items && answer.items.length > 0 && (
          <ResultTable items={answer.items} />
        )}
        <AssetGallery assets={answer?.assets || []} />
        <SourceCards sources={answer?.sources || []} />
        {response?.trace_id && (
          <div className="mono mt-4 rounded-lg bg-slate-100 px-3 py-2 text-[10px] text-slate-500">
            trace_id: {response.trace_id}
          </div>
        )}
        {!message.error && !error && (
          <div className="mt-4 flex items-center gap-1 border-t border-slate-100 pt-3">
            <button
              aria-label="Helpful response"
              className="rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-emerald-600"
            >
              <ThumbsUp size={14} />
            </button>
            <button
              aria-label="Unhelpful response"
              className="rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-rose-600"
            >
              <ThumbsDown size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function ResultTable({ items }: { items: Array<Record<string, unknown>> }) {
  const normalized = items.filter((item) => item.data && typeof item.data === "object");
  if (!normalized.length) return null;
  const columns = Array.from(
    new Set(
      normalized.flatMap((item) => Object.keys(item.data as Record<string, unknown>)),
    ),
  ).slice(0, 6);
  if (!columns.length) return null;
  return (
    <div className="mt-4 overflow-auto rounded-xl border border-slate-200">
      <table className="w-full min-w-[520px] text-left text-xs">
        <thead className="bg-slate-50">
          <tr>
            {columns.map((column) => (
              <th key={column} className="px-3 py-2 font-bold text-slate-500">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {normalized.map((item, index) => (
            <tr key={String(item.id || index)} className="border-t border-slate-100">
              {columns.map((column) => (
                <td key={column} className="max-w-48 truncate px-3 py-2">
                  {String((item.data as Record<string, unknown>)[column] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
