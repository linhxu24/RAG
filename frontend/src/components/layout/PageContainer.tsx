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
    <div className="mx-auto w-full max-w-[1920px] px-3.5 py-4 md:px-5">
      <div className="mb-4 flex flex-col items-start justify-between gap-3 lg:flex-row lg:gap-5">
        <div>
          <h1 className="text-[22px] font-bold tracking-tight text-slate-950">{title}</h1>
          <p className="mt-0.5 text-xs leading-5 text-slate-500">{description}</p>
        </div>
        {actions && <div className="w-full overflow-x-auto lg:w-auto">{actions}</div>}
      </div>
      {children}
    </div>
  );
}
