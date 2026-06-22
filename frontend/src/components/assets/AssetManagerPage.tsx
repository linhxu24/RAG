import { useQuery } from "@tanstack/react-query";
import { Eye, ImageOff, X } from "lucide-react";
import { useState } from "react";

import { getAsset, listAssets } from "../../api/assets";
import type { AssetRecord } from "../../types";
import { assetUrl, truncate } from "../../utils/format";
import { DataTable, type Column } from "../common/DataTable";
import { JsonViewer } from "../common/JsonViewer";
import { ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";

export function AssetManagerPage() {
  const [selected, setSelected] = useState<string>();
  const assets = useQuery({ queryKey: ["assets"], queryFn: listAssets });
  const detail = useQuery({
    queryKey: ["asset", selected],
    queryFn: () => getAsset(selected!),
    enabled: Boolean(selected),
  });
  const columns: Column<AssetRecord>[] = [
    {
      key: "preview",
      label: "preview",
      render: (row) => <AssetThumb asset={row} />,
      sortable: false,
    },
    {
      key: "id",
      label: "asset_id",
      render: (row) => <span className="mono text-[10px]">{truncate(row.asset_id, 18)}</span>,
      sortValue: (row) => row.asset_id,
      searchValue: (row) => row.asset_id,
    },
    {
      key: "token",
      label: "asset_token",
      render: (row) => <span className="mono text-[10px]">{row.asset_token || "—"}</span>,
      sortValue: (row) => row.asset_token,
      searchValue: (row) => row.asset_token,
    },
    {
      key: "type",
      label: "type",
      render: (row) => row.asset_type || "—",
      sortValue: (row) => row.asset_type,
      searchValue: (row) => row.asset_type,
    },
    {
      key: "doc",
      label: "doc_id",
      render: (row) => <span className="mono text-[10px]">{truncate(row.doc_id, 16)}</span>,
      sortValue: (row) => row.doc_id,
      searchValue: (row) => row.doc_id,
    },
    {
      key: "status",
      label: "status",
      render: (row) => <StatusBadge status={row.status} />,
      sortValue: (row) => row.status,
      searchValue: (row) => row.status,
    },
    {
      key: "action",
      label: "action",
      sortable: false,
      render: (row) => (
        <button
          className="secondary-button p-2"
          title="View asset"
          onClick={() => setSelected(row.asset_id)}
        >
          <Eye size={14} />
        </button>
      ),
    },
  ];
  return (
    <PageContainer
      title="Asset Manager"
      description="Kiểm tra token mapping, preview ảnh, đường dẫn local/public và nơi asset được sử dụng."
    >
      {assets.isLoading ? (
        <LoadingState />
      ) : assets.isError ? (
        <ErrorState error={assets.error} onRetry={() => void assets.refetch()} />
      ) : (
        <DataTable
          rows={assets.data?.items || []}
          columns={columns}
          rowKey={(row) => row.asset_id}
          emptyTitle="Chưa có asset"
          defaultSort={{ key: "id", direction: "desc" }}
        />
      )}
      {selected && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/30 p-8 backdrop-blur-sm">
          <div className="panel max-h-[88vh] w-full max-w-4xl overflow-y-auto">
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">
              <div>
                <div className="font-bold text-slate-900">Asset detail</div>
                <div className="mono text-[10px] text-slate-400">{selected}</div>
              </div>
              <button onClick={() => setSelected(undefined)}>
                <X size={19} className="text-slate-500" />
              </button>
            </div>
            <div className="p-6">
              {detail.isLoading ? (
                <LoadingState />
              ) : detail.isError ? (
                <ErrorState error={detail.error} />
              ) : detail.data ? (
                <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_1fr]">
                  <div className="panel overflow-hidden">
                    <AssetPreview asset={detail.data} />
                  </div>
                  <div className="space-y-4">
                    <div className="panel p-5">
                      <StatusBadge status={detail.data.status} />
                      <div className="mono mt-4 break-all rounded-xl bg-slate-50 p-3 text-xs">
                        {detail.data.asset_token}
                      </div>
                      <dl className="mt-4 space-y-3 text-xs">
                        {[
                          ["Document", detail.data.doc_id],
                          ["Chunk", detail.data.chunk_id],
                          ["Local path", detail.data.local_path],
                          ["Public URL", detail.data.public_url],
                        ].map(([label, value]) => (
                          <div key={label}>
                            <dt className="font-bold uppercase tracking-wider text-slate-400">
                              {label}
                            </dt>
                            <dd className="mono mt-1 break-all text-slate-700">{value || "—"}</dd>
                          </div>
                        ))}
                      </dl>
                    </div>
                    <div className="panel p-5">
                      <div className="mb-2 text-sm font-bold text-slate-800">Used in</div>
                      <JsonViewer value={detail.data.used_in || {}} maxHeight="280px" />
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </PageContainer>
  );
}

function AssetThumb({ asset }: { asset: AssetRecord }) {
  const [broken, setBroken] = useState(false);
  const source = assetUrl(asset.public_url);
  return source && !broken ? (
    <img
      src={source}
      alt={asset.asset_token}
      onError={() => setBroken(true)}
      className="h-11 w-14 rounded-lg object-cover"
    />
  ) : (
    <span className="grid h-11 w-14 place-items-center rounded-lg bg-slate-100 text-slate-400">
      <ImageOff size={17} />
    </span>
  );
}

function AssetPreview({ asset }: { asset: AssetRecord }) {
  const [broken, setBroken] = useState(false);
  const source = assetUrl(asset.public_url);
  return source && !broken ? (
    <img
      src={source}
      alt={asset.asset_token}
      onError={() => setBroken(true)}
      className="min-h-[420px] w-full bg-slate-100 object-contain"
    />
  ) : (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-3 bg-slate-100 text-slate-500">
      <ImageOff size={30} />
      <span className="text-sm font-bold">Asset preview unavailable</span>
    </div>
  );
}
