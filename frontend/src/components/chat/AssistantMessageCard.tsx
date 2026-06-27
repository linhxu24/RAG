import { AlertTriangle, Bot, Sparkles } from "lucide-react";

import type { ChatMessage, ChatSuggestion } from "../../types";
import {
  formatBoolean,
  formatCurrency,
  formatDuration,
  formatNumber,
  formatUnknownValue,
} from "../../utils/format";
import { AssetGallery } from "./AssetGallery";
import { SourceCards } from "./SourceCards";

export function AssistantMessageCard({
  message,
  onSuggestion,
  disabled = false,
}: {
  message: ChatMessage;
  onSuggestion?: (suggestion: ChatSuggestion) => void;
  disabled?: boolean;
}) {
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
        {response?.suggestions && response.suggestions.length > 0 && (
          <div className="mt-4 border-t border-slate-100 pt-4">
            <div className="mb-2 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
              <Sparkles size={13} className="text-teal-600" />
              Bạn có thể hỏi tiếp
            </div>
            <div className="flex flex-wrap gap-2">
              {response.suggestions.map((suggestion) => (
                <button
                  key={suggestion.suggestion_id}
                  type="button"
                  disabled={disabled}
                  onClick={() => onSuggestion?.(suggestion)}
                  title={suggestion.query}
                  className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-left text-xs font-semibold text-teal-800 transition hover:border-teal-300 hover:bg-teal-100 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {suggestion.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {response?.trace_id && (
          <div className="mono mt-4 rounded-lg bg-slate-100 px-3 py-2 text-[10px] text-slate-500">
            trace_id: {response.trace_id}
          </div>
        )}
      </div>
    </div>
  );
}

function ResultTable({ items }: { items: Array<Record<string, unknown>> }) {
  const normalized = items.filter(
    (item) => item.data && typeof item.data === "object" && !Array.isArray(item.data),
  );
  if (!normalized.length) return null;
  const sourceTypes = new Set(normalized.map((item) => String(item.type || "unknown")));
  const preferred = preferredColumns(sourceTypes);
  const available = new Set(
    normalized.flatMap((item) => Object.keys(item.data as Record<string, unknown>)),
  );
  const columns = preferred.filter((key) => available.has(key));
  if (!columns.length) return null;
  return (
    <div className="mt-4 overflow-auto rounded-xl border border-slate-200">
      <table className="w-full min-w-[520px] text-left text-xs">
        <thead className="bg-slate-50">
          <tr>
            {columns.map((column) => (
              <th key={column} className="px-3 py-2 font-bold text-slate-500">
                {columnLabel(column)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {normalized.map((item, index) => (
            <tr key={String(item.id || index)} className="border-t border-slate-100">
              {columns.map((column) => (
                <td key={column} className="max-w-64 px-3 py-2 align-top">
                  <ResultValue
                    column={column}
                    data={item.data as Record<string, unknown>}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function preferredColumns(sourceTypes: Set<string>): string[] {
  if (sourceTypes.size === 1 && sourceTypes.has("product")) {
    return ["name", "category", "brand", "model", "price", "quantity", "description"];
  }
  if (sourceTypes.size === 1 && sourceTypes.has("service")) {
    return [
      "name",
      "source_category",
      "duration_minutes",
      "price",
      "description",
      "indications",
    ];
  }
  if (sourceTypes.size === 1 && sourceTypes.has("faq")) {
    return ["question", "answer", "category"];
  }
  if (sourceTypes.size === 1 && sourceTypes.has("clinic_info")) {
    return ["key", "value"];
  }
  return ["name", "question", "category", "source_category", "price", "description", "value"];
}

function columnLabel(column: string): string {
  const labels: Record<string, string> = {
    name: "Tên",
    question: "Câu hỏi",
    answer: "Trả lời",
    category: "Danh mục",
    source_category: "Danh mục nguồn",
    brand: "Thương hiệu",
    model: "Mẫu",
    price: "Giá",
    quantity: "Số lượng",
    duration_minutes: "Thời lượng",
    description: "Mô tả",
    indications: "Chỉ định",
    key: "Thông tin",
    value: "Giá trị",
  };
  return labels[column] || column.replaceAll("_", " ");
}

function ResultValue({
  column,
  data,
}: {
  column: string;
  data: Record<string, unknown>;
}) {
  const value = data[column];
  if (column === "price") {
    return (
      <span className="whitespace-nowrap font-semibold text-slate-800">
        {formatCurrency(toNumber(value), String(data.currency || "VND"))}
      </span>
    );
  }
  if (column === "duration_minutes") {
    return <span className="whitespace-nowrap">{formatDuration(toNumber(value))}</span>;
  }
  if (column === "quantity") {
    const quantity = toNumber(value);
    return quantity == null ? "—" : formatNumber(quantity);
  }
  if (typeof value === "boolean") return formatBoolean(value);
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return (
      <details>
        <summary className="cursor-pointer font-semibold text-teal-700">Xem chi tiết</summary>
        <pre className="mono mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-950 p-2 text-[10px] text-slate-200">
          {JSON.stringify(value, null, 2)}
        </pre>
      </details>
    );
  }
  const text = formatUnknownValue(value);
  return (
    <span className={column === "description" || column === "answer" ? "line-clamp-3" : ""} title={text}>
      {text}
    </span>
  );
}

function toNumber(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
