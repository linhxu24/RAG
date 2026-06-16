import type { LucideIcon } from "lucide-react";

export function MetricCard({
  label,
  value,
  hint,
  icon: Icon,
  accent = "teal",
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon?: LucideIcon;
  accent?: "teal" | "blue" | "amber" | "rose" | "violet";
}) {
  const accents = {
    teal: "bg-teal-50 text-teal-700",
    blue: "bg-blue-50 text-blue-700",
    amber: "bg-amber-50 text-amber-700",
    rose: "bg-rose-50 text-rose-700",
    violet: "bg-violet-50 text-violet-700",
  };
  return (
    <div className="panel flex min-h-32 flex-col justify-between p-5">
      <div className="flex items-start justify-between gap-3">
        <span className="text-sm font-semibold text-slate-500">{label}</span>
        {Icon && (
          <span className={`rounded-xl p-2.5 ${accents[accent]}`}>
            <Icon size={18} />
          </span>
        )}
      </div>
      <div>
        <div className="text-2xl font-bold tracking-tight text-slate-900">{value}</div>
        {hint && <div className="mt-1 text-xs text-slate-400">{hint}</div>}
      </div>
    </div>
  );
}
