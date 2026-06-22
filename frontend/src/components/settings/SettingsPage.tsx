import { useQuery } from "@tanstack/react-query";
import { Cpu, Database, KeyRound, Settings2 } from "lucide-react";

import { API_BASE_URL, PUBLIC_ASSETS_BASE_URL } from "../../api/client";
import { getSettings } from "../../api/settings";
import { ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";

export function SettingsPage() {
  const query = useQuery({ queryKey: ["settings"], queryFn: getSettings });
  return (
    <PageContainer
      title="Settings"
      description="Frontend environment và backend RAG configuration. Backend settings hiện read-only."
    >
      <div className="mb-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
        <EnvironmentCard
          title="API Base URL"
          value={API_BASE_URL}
          icon={Database}
        />
        <EnvironmentCard
          title="Public Assets Base URL"
          value={PUBLIC_ASSETS_BASE_URL}
          icon={KeyRound}
        />
      </div>
      {query.isLoading ? (
        <LoadingState />
      ) : query.isError ? (
        <ErrorState error={query.error} onRetry={() => void query.refetch()} />
      ) : query.data ? (
        <div className="panel overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
            <div className="flex items-center gap-2 font-bold text-slate-800">
              <Settings2 size={17} className="text-teal-600" /> Backend configuration
            </div>
            <StatusBadge status={query.data.read_only ? "read only" : "editable"} />
          </div>
          <div className="grid grid-cols-1 gap-x-8 px-5 py-2 lg:grid-cols-2">
            {Object.entries(query.data)
              .filter(([key]) => key !== "read_only")
              .map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between gap-6 border-b border-slate-100 py-4"
                >
                  <div className="text-xs font-bold text-slate-500">
                    {key.replaceAll("_", " ")}
                  </div>
                  {typeof value === "boolean" ? (
                    <StatusBadge status={value ? "enabled" : "disabled"} />
                  ) : (
                    <div className="mono max-w-[60%] break-all text-right text-xs font-semibold text-slate-800">
                      {String(value)}
                    </div>
                  )}
                </div>
              ))}
          </div>
          <div className="border-t border-slate-200 bg-slate-50 px-5 py-4 text-xs leading-5 text-slate-500">
            Thay đổi runtime settings qua file <code className="mono">.env</code> và restart
            backend. UI không ghi secret hoặc database credential.
          </div>
        </div>
      ) : null}
    </PageContainer>
  );
}

function EnvironmentCard({
  title,
  value,
  icon: Icon,
}: {
  title: string;
  value: string;
  icon: typeof Cpu;
}) {
  return (
    <div className="panel flex items-center gap-4 p-5">
      <span className="rounded-xl bg-teal-50 p-3 text-teal-700">
        <Icon size={20} />
      </span>
      <div className="min-w-0">
        <div className="text-xs font-bold text-slate-500">{title}</div>
        <div className="mono mt-1 truncate text-sm font-semibold text-slate-800">{value}</div>
      </div>
    </div>
  );
}
