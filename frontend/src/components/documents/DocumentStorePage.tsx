import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  CheckCircle2,
  Eye,
  RefreshCw,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import { useState } from "react";

import {
  activateDocument,
  archiveDocument,
  deleteDocument,
  getDocument,
  listDocuments,
  reingestDocument,
  resetApplicationData,
} from "../../api/documents";
import type { DocumentRecord } from "../../types";
import { formatDate, truncate } from "../../utils/format";
import { DataTable, type Column } from "../common/DataTable";
import { JsonViewer } from "../common/JsonViewer";
import { ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { useToast } from "../common/Toast";
import { PageContainer } from "../layout/PageContainer";

export function DocumentStorePage() {
  const queryClient = useQueryClient();
  const notify = useToast();
  const [selected, setSelected] = useState<string>();
  const query = useQuery({ queryKey: ["documents"], queryFn: listDocuments });
  const detail = useQuery({
    queryKey: ["document", selected],
    queryFn: () => getDocument(selected!),
    enabled: Boolean(selected),
  });
  const action = useMutation({
    mutationFn: async ({ type, id }: { type: string; id: string }) => {
      if (type === "activate") return activateDocument(id);
      if (type === "archive") return archiveDocument(id);
      if (type === "delete") {
        if (!window.confirm("Xóa vĩnh viễn document và toàn bộ dữ liệu liên quan?")) {
          throw new Error("Action cancelled");
        }
        return deleteDocument(id);
      }
      return reingestDocument(id);
    },
    onSuccess: async () => {
      notify("Document action completed.");
      await queryClient.invalidateQueries({ queryKey: ["documents"] });
      if (selected) await queryClient.invalidateQueries({ queryKey: ["document", selected] });
    },
    onError: (error) => notify(error.message, "error"),
  });

  const columns: Column<DocumentRecord>[] = [
    {
      key: "id",
      label: "doc_id",
      render: (row) => <span className="mono text-[10px]">{truncate(row.doc_id, 16)}</span>,
    },
    {
      key: "file",
      label: "file_name",
      render: (row) => <span className="font-semibold text-slate-800">{row.file_name}</span>,
    },
    { key: "type", label: "type", render: (row) => row.file_type || "—" },
    { key: "status", label: "status", render: (row) => <StatusBadge status={row.status} /> },
    { key: "version", label: "version", render: (row) => row.version },
    { key: "chunks", label: "chunks", render: (row) => row.chunks },
    { key: "tables", label: "tables", render: (row) => row.tables },
    { key: "assets", label: "assets", render: (row) => row.assets },
    { key: "created", label: "created_at", render: (row) => formatDate(row.created_at) },
    {
      key: "actions",
      label: "actions",
      render: (row) => (
        <div className="flex items-center gap-1">
          <ActionIcon label="View" icon={Eye} onClick={() => setSelected(row.doc_id)} />
          <ActionIcon
            label="Activate"
            icon={CheckCircle2}
            onClick={() => action.mutate({ type: "activate", id: row.doc_id })}
          />
          <ActionIcon
            label="Archive"
            icon={Archive}
            onClick={() => action.mutate({ type: "archive", id: row.doc_id })}
          />
          <ActionIcon
            label="Re-ingest"
            icon={RotateCcw}
            onClick={() => action.mutate({ type: "reingest", id: row.doc_id })}
          />
          <ActionIcon
            label="Delete permanently"
            icon={Trash2}
            onClick={() => action.mutate({ type: "delete", id: row.doc_id })}
          />
        </div>
      ),
    },
  ];

  return (
    <PageContainer
      title="Document Store"
      description="Quản lý document lifecycle, metadata và các artifact được tạo trong ingestion."
      actions={
        <div className="flex gap-2">
          <button
            className="secondary-button px-3 py-2 text-xs"
            onClick={() => void query.refetch()}
          >
            <RefreshCw size={14} className={query.isFetching ? "animate-spin" : ""} />
            Refresh
          </button>
          <button
            className="secondary-button border-red-200 px-3 py-2 text-xs text-red-700"
            onClick={async () => {
              const confirmation = window.prompt(
                "Nhập DELETE SIMPLYDENT CONTENT để xóa toàn bộ content đã ingest.",
              );
              if (!confirmation) return;
              try {
                await resetApplicationData("content", confirmation);
                notify("Đã xóa content; taxonomy vẫn được giữ lại.");
                await queryClient.invalidateQueries();
              } catch (error) {
                notify(
                  error instanceof Error ? error.message : "Reset failed",
                  "error",
                );
              }
            }}
          >
            <Trash2 size={14} /> Reset content
          </button>
        </div>
      }
    >
      {query.isLoading ? (
        <LoadingState />
      ) : query.isError ? (
        <ErrorState error={query.error} onRetry={() => void query.refetch()} />
      ) : (
        <DataTable
          rows={query.data?.items || []}
          columns={columns}
          rowKey={(row) => row.doc_id}
          emptyTitle="Chưa có document"
        />
      )}
      {selected && (
        <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/25 backdrop-blur-sm">
          <div className="h-full w-[720px] overflow-y-auto border-l border-slate-200 bg-[#f7f9fc] shadow-2xl">
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
              <div>
                <div className="font-bold text-slate-900">Document detail</div>
                <div className="mono mt-1 text-[10px] text-slate-400">{selected}</div>
              </div>
              <button
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-100"
                onClick={() => setSelected(undefined)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="space-y-5 p-6">
              {detail.isLoading ? (
                <LoadingState />
              ) : detail.isError ? (
                <ErrorState error={detail.error} />
              ) : detail.data ? (
                <DocumentDetail document={detail.data} />
              ) : null}
            </div>
          </div>
        </div>
      )}
    </PageContainer>
  );
}

function ActionIcon({
  label,
  icon: Icon,
  onClick,
}: {
  label: string;
  icon: typeof Eye;
  onClick: () => void;
}) {
  return (
    <button
      title={label}
      className="rounded-lg border border-slate-200 p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-800"
      onClick={onClick}
    >
      <Icon size={14} />
    </button>
  );
}

function DocumentDetail({ document }: { document: DocumentRecord }) {
  const sections = [
    ["Metadata", document.metadata],
    ["Ingestion quality report", document.latest_ingestion_run?.quality_report],
    ["Chunks", document.chunk_items],
    ["Tables", document.table_items],
    ["Assets", document.asset_items],
  ] as const;
  return (
    <>
      <div className="panel p-5">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-bold text-slate-900">{document.file_name}</h2>
            <div className="mt-1 text-xs text-slate-500">
              {document.file_type} · version {document.version}
            </div>
          </div>
          <StatusBadge status={document.status} />
        </div>
        <div className="mt-3 text-xs text-slate-500">
          Auto detect:{" "}
          <span className="font-semibold text-slate-700">
            {document.detected_document_type || "unknown"}
          </span>
          {document.document_type_confidence != null
            ? ` (${document.document_type_confidence.toFixed(2)})`
            : ""}
        </div>
        <div className="mt-5 grid grid-cols-3 gap-3">
          {[
            ["Chunks", document.chunks],
            ["Tables", document.tables],
            ["Assets", document.assets],
          ].map(([label, value]) => (
            <div key={label} className="rounded-xl bg-slate-50 p-3 text-center">
              <div className="text-xl font-bold text-slate-900">{value}</div>
              <div className="text-[10px] font-bold uppercase text-slate-400">{label}</div>
            </div>
          ))}
        </div>
      </div>
      {sections.map(([title, value]) => (
        <details key={title} className="panel overflow-hidden" open={title === "Metadata"}>
          <summary className="cursor-pointer px-5 py-4 text-sm font-bold text-slate-800">
            {title}
          </summary>
          <div className="border-t border-slate-100 p-4">
            <JsonViewer value={value || {}} />
          </div>
        </details>
      ))}
    </>
  );
}
