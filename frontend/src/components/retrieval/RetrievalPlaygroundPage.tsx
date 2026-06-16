import { useMutation } from "@tanstack/react-query";
import {
  Binary,
  Braces,
  Database,
  Filter,
  Layers3,
  Search,
  Sparkles,
} from "lucide-react";
import { useState } from "react";

import {
  runRetrievalDebug,
  type RetrievalOptions,
} from "../../api/retrieval";
import type { RetrievalItem } from "../../types";
import { JsonViewer } from "../common/JsonViewer";
import { EmptyState, ErrorState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";

const tabs = [
  "Router",
  "Structured",
  "Dense",
  "Sparse",
  "RRF",
  "Reranker",
  "Final Context",
] as const;
type Tab = (typeof tabs)[number];

export function RetrievalPlaygroundPage() {
  const [activeTab, setActiveTab] = useState<Tab>("Router");
  const [options, setOptions] = useState<RetrievalOptions>({
    query: "",
    use_structured: true,
    use_dense: true,
    use_sparse: true,
    use_rrf: true,
    use_reranker: false,
    use_hyde: false,
  });
  const mutation = useMutation({ mutationFn: runRetrievalDebug });
  const run = () => {
    if (options.query.trim()) mutation.mutate(options);
  };
  return (
    <PageContainer
      title="Retrieval Playground"
      description="Chạy từng nhánh structured, dense, sparse, RRF, reranker và xem final context trước generation."
    >
      <div className="panel p-5">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
              size={18}
            />
            <input
              className="control py-3 pl-10 pr-4 text-sm"
              placeholder="Nhập query cần debug retrieval..."
              value={options.query}
              onChange={(event) => setOptions({ ...options, query: event.target.value })}
              onKeyDown={(event) => event.key === "Enter" && run()}
            />
          </div>
          <button
            className="primary-button px-5 py-3 text-sm"
            disabled={!options.query.trim() || mutation.isPending}
            onClick={run}
          >
            <Search size={17} />
            {mutation.isPending ? "Đang chạy..." : "Run Retrieval"}
          </button>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {(
            [
            ["use_structured", "Structured SQL", Database],
            ["use_dense", "Dense Retrieval", Binary],
            ["use_sparse", "Sparse Retrieval", Filter],
            ["use_rrf", "RRF", Layers3],
            ["use_reranker", "Reranker", Sparkles],
            ["use_hyde", "HyDE", Braces],
            ] as const
          ).map(([key, label, Icon]) => (
            <label
              key={String(key)}
              className={`flex cursor-pointer items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold transition ${
                options[key as keyof RetrievalOptions]
                  ? "border-teal-200 bg-teal-50 text-teal-800"
                  : "border-slate-200 bg-white text-slate-500"
              }`}
            >
              <input
                type="checkbox"
                className="hidden"
                checked={Boolean(options[key])}
                onChange={(event) =>
                  setOptions({ ...options, [key]: event.target.checked })
                }
              />
              <Icon size={14} /> {String(label)}
            </label>
          ))}
        </div>
      </div>
      <div className="mt-5">
        <div className="mb-3 flex gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-white p-1.5">
          {tabs.map((tab) => (
            <button
              key={tab}
              className={`rounded-lg px-4 py-2 text-xs font-bold ${
                activeTab === tab
                  ? "bg-[#10253f] text-white"
                  : "text-slate-500 hover:bg-slate-100"
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
        {mutation.isError ? (
          <ErrorState error={mutation.error} onRetry={run} />
        ) : !mutation.data ? (
          <EmptyState
            title="Chưa chạy retrieval"
            description="Nhập query và chọn các retriever để xem kết quả từng stage."
          />
        ) : (
          <RetrievalTab active={activeTab} data={mutation.data} />
        )}
      </div>
    </PageContainer>
  );
}

function RetrievalTab({
  active,
  data,
}: {
  active: Tab;
  data: Awaited<ReturnType<typeof runRetrievalDebug>>;
}) {
  if (active === "Router") {
    return (
      <div className="grid grid-cols-[340px_1fr] gap-4">
        <div className="panel p-5">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Detected intent
          </div>
          <div className="mt-3 text-2xl font-bold text-slate-900">
            {String(data.router.intent)}
          </div>
          <div className="mt-3 flex gap-2">
            <StatusBadge status={data.router.needs_rag ? "RAG required" : "Direct route"} />
            <StatusBadge
              status={data.router.needs_clarification ? "Needs clarification" : "Confident"}
            />
          </div>
          <div className="mt-5 text-sm text-slate-500">
            Confidence:{" "}
            <strong className="text-slate-800">
              {(Number(data.router.confidence || 0) * 100).toFixed(1)}%
            </strong>
          </div>
        </div>
        <div className="panel p-5">
          <JsonViewer
            value={{
              ...data.router,
              entities: data.entities,
              plan: data.plan,
              rewrite: data.rewrite,
            }}
          />
        </div>
      </div>
    );
  }
  if (active === "Final Context") {
    return (
      <div className="panel p-5">
        <div className="mb-3 text-sm font-bold text-slate-800">
          Final context · {data.final_context.total_chars} characters
        </div>
        <JsonViewer value={data.final_context} maxHeight="560px" />
      </div>
    );
  }
  const key = active.toLowerCase() as
    | "structured"
    | "dense"
    | "sparse"
    | "rrf"
    | "reranker";
  return <ResultList items={data[key]} />;
}

function ResultList({ items }: { items: RetrievalItem[] }) {
  if (!items.length) return <EmptyState title="Không có retrieval result" />;
  return (
    <div className="space-y-3">
      {items.map((item) => (
        <details key={`${item.type}-${item.id}`} className="panel overflow-hidden">
          <summary className="cursor-pointer list-none p-4">
            <div className="flex items-start gap-4">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-slate-100 text-sm font-bold text-slate-700">
                {item.rank}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <StatusBadge status={item.type} compact />
                  <span className="mono truncate text-[10px] text-slate-400">
                    {item.id}
                  </span>
                </div>
                <p className="line-clamp-3 mt-2 text-sm leading-6 text-slate-600">
                  {item.content}
                </p>
              </div>
              <div className="rounded-lg bg-teal-50 px-3 py-2 text-xs font-bold text-teal-700">
                {Number(item.score).toFixed(4)}
              </div>
            </div>
          </summary>
          <div className="grid grid-cols-2 gap-3 border-t border-slate-100 bg-slate-50 p-4">
            <JsonViewer value={item.source || {}} maxHeight="260px" />
            <JsonViewer value={item.metadata || {}} maxHeight="260px" />
          </div>
        </details>
      ))}
    </div>
  );
}
