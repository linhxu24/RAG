import clsx from "clsx";

import type { StatusValue } from "../../types";

const positive = new Set([
  "success",
  "active",
  "completed",
  "connected",
  "enabled",
  "available",
  "configured",
  "ok",
  "ready",
]);
const negative = new Set(["failed", "disconnected", "missing", "error"]);
const warning = new Set(["running", "pending", "review_required", "degraded"]);

export function StatusBadge({
  status = "unknown",
  compact = false,
}: {
  status?: StatusValue;
  compact?: boolean;
}) {
  const normalized = String(status).toLowerCase();
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border font-semibold capitalize",
        compact ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs",
        positive.has(normalized) &&
          "border-emerald-200 bg-emerald-50 text-emerald-700",
        negative.has(normalized) && "border-rose-200 bg-rose-50 text-rose-700",
        warning.has(normalized) && "border-amber-200 bg-amber-50 text-amber-700",
        !positive.has(normalized) &&
          !negative.has(normalized) &&
          !warning.has(normalized) &&
          "border-slate-200 bg-slate-50 text-slate-600",
      )}
    >
      <span
        className={clsx(
          "h-1.5 w-1.5 rounded-full",
          positive.has(normalized) && "bg-emerald-500",
          negative.has(normalized) && "bg-rose-500",
          warning.has(normalized) && "bg-amber-500",
          !positive.has(normalized) &&
            !negative.has(normalized) &&
            !warning.has(normalized) &&
            "bg-slate-400",
        )}
      />
      {String(status).replaceAll("_", " ")}
    </span>
  );
}
