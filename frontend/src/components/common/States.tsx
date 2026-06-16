import {
  AlertTriangle,
  FileQuestion,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";

import { endpointUnavailable } from "../../api/client";

export function LoadingState({ label = "Đang tải dữ liệu..." }: { label?: string }) {
  return (
    <div className="panel flex min-h-44 items-center justify-center gap-3 p-8 text-sm text-slate-500">
      <LoaderCircle className="animate-spin text-teal-600" size={20} />
      {label}
    </div>
  );
}

export function EmptyState({
  title = "Chưa có dữ liệu",
  description = "Dữ liệu sẽ xuất hiện tại đây khi backend có bản ghi.",
}: {
  title?: string;
  description?: string;
}) {
  return (
    <div className="panel flex min-h-44 flex-col items-center justify-center px-8 py-12 text-center">
      <span className="mb-4 rounded-2xl bg-slate-100 p-3 text-slate-500">
        <FileQuestion size={24} />
      </span>
      <div className="font-bold text-slate-800">{title}</div>
      <p className="mt-1 max-w-md text-sm leading-6 text-slate-500">{description}</p>
    </div>
  );
}

export function ErrorState({
  error,
  onRetry,
  title = "Không thể tải dữ liệu",
}: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  return (
    <div className="panel flex min-h-44 flex-col items-center justify-center border-rose-100 bg-rose-50/40 px-8 py-10 text-center">
      <span className="mb-3 rounded-2xl bg-rose-100 p-3 text-rose-600">
        <AlertTriangle size={23} />
      </span>
      <div className="font-bold text-slate-800">{title}</div>
      <p className="mt-1 max-w-xl text-sm leading-6 text-slate-600">
        {endpointUnavailable(error)}
      </p>
      {onRetry && (
        <button className="secondary-button mt-4 px-3 py-2 text-xs" onClick={onRetry}>
          <RefreshCw size={14} /> Thử lại
        </button>
      )}
    </div>
  );
}
