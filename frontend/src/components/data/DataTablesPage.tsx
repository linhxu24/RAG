import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import {
  getDataTable,
  type DataTableName,
} from "../../api/dataTables";
import { truncate } from "../../utils/format";
import { DataTable, type Column } from "../common/DataTable";
import { ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";

const tabs: Array<{ key: DataTableName; label: string }> = [
  { key: "products", label: "Products" },
  { key: "services", label: "Services" },
  { key: "faqs", label: "FAQs" },
  { key: "clinic-info", label: "Clinic Info" },
  { key: "tables", label: "Tables" },
  { key: "table-rows", label: "Table Rows" },
  { key: "chunks", label: "Chunks" },
];

export function DataTablesPage() {
  const [active, setActive] = useState<DataTableName>("products");
  const query = useQuery({
    queryKey: ["data-table", active],
    queryFn: () => getDataTable(active),
  });
  const rows = query.data?.items || [];
  return (
    <PageContainer
      title="Data Tables"
      description="Inspect normalized business data, parsed tables, row records và text chunks."
    >
      <div className="mb-4 flex gap-1 rounded-xl border border-slate-200 bg-white p-1.5">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={`rounded-lg px-4 py-2 text-xs font-bold ${
              active === tab.key
                ? "bg-[#10253f] text-white"
                : "text-slate-500 hover:bg-slate-100"
            }`}
            onClick={() => setActive(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {query.isLoading ? (
        <LoadingState />
      ) : query.isError ? (
        <ErrorState error={query.error} onRetry={() => void query.refetch()} />
      ) : (
        <DynamicTable name={active} rows={rows} />
      )}
    </PageContainer>
  );
}

function DynamicTable({
  name,
  rows,
}: {
  name: DataTableName;
  rows: Array<Record<string, any>>;
}) {
  const preferred: Record<DataTableName, string[]> = {
    products: [
      "name",
      "category",
      "description",
      "price",
      "quantity",
      "asset_id",
      "source_doc_id",
      "status",
      "version",
    ],
    services: [
      "name",
      "description",
      "duration_minutes",
      "price",
      "source_doc_id",
      "status",
      "version",
    ],
    faqs: ["question", "answer", "category", "is_active", "embedding_status"],
    "clinic-info": ["key", "value", "status", "source_doc_id"],
    tables: ["table_id", "doc_id", "table_name", "page_number", "status"],
    "table-rows": [
      "row_id",
      "table_id",
      "entity_type",
      "entity_name",
      "row_text",
      "row_json",
      "embedding_status",
    ],
    chunks: [
      "chunk_id",
      "doc_id",
      "chunk_index",
      "content",
      "page_number",
      "embedding_status",
      "status",
    ],
  };
  const keys = preferred[name];
  const columns: Column<Record<string, any>>[] = keys.map((key) => ({
    key,
    label: key,
    render: (row) => {
      const value = row[key];
      if (key === "status" || key === "embedding_status") {
        return <StatusBadge status={String(value || "unknown")} />;
      }
      if (typeof value === "object" && value !== null) {
        return (
          <details>
            <summary className="cursor-pointer text-xs font-semibold text-teal-700">
              View JSON
            </summary>
            <pre className="mono mt-2 max-h-48 overflow-auto rounded-lg bg-slate-950 p-2 text-[10px] text-slate-200">
              {JSON.stringify(value, null, 2)}
            </pre>
          </details>
        );
      }
      return (
        <span
          className={key.includes("id") ? "mono text-[10px]" : "line-clamp-3 text-xs"}
          title={String(value ?? "")}
        >
          {truncate(value == null ? "—" : String(value), key.includes("id") ? 20 : 100)}
        </span>
      );
    },
  }));
  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(row) =>
        String(
          row.product_id ||
            row.service_id ||
            row.faq_id ||
            row.id ||
            row.table_id ||
            row.row_id ||
            row.chunk_id ||
            JSON.stringify(row),
        )
      }
      emptyTitle={`No ${name} data`}
    />
  );
}
