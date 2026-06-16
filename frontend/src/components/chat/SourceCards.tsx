import { FileText } from "lucide-react";

import type { SourceRecord } from "../../types";

export function SourceCards({ sources }: { sources: SourceRecord[] }) {
  if (!sources.length) return null;
  return (
    <div className="mt-4 space-y-2">
      <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
        Sources
      </div>
      {sources.map((source, index) => (
        <div
          key={`${source.source_id}-${index}`}
          className="flex items-start gap-2.5 rounded-xl border border-slate-200 bg-slate-50 p-3"
        >
          <FileText size={16} className="mt-0.5 shrink-0 text-teal-600" />
          <div className="min-w-0">
            <div className="text-xs font-bold text-slate-700">
              {source.file_name || source.source_type || "Retrieved source"}
              {source.page_number != null ? ` — page ${source.page_number}` : ""}
            </div>
            <div className="mono mt-1 truncate text-[10px] text-slate-400">
              {source.chunk_id || source.row_id || source.source_id || source.doc_id}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
