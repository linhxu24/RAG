import type { ReactNode } from "react";

export function PageContainer({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto w-full max-w-[1680px] px-6 py-6">
      <div className="mb-6 flex items-start justify-between gap-6">
        <div>
          <h1 className="text-[28px] font-bold tracking-tight text-slate-950">{title}</h1>
          <p className="mt-1 text-sm leading-6 text-slate-500">{description}</p>
        </div>
        {actions}
      </div>
      {children}
    </div>
  );
}
