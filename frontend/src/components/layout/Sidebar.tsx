import { Boxes, Menu, Settings, X } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";

import {
  navigationGroups,
  navigationItemForPath,
} from "./navigation";

export function Sidebar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const current = navigationItemForPath(location.pathname);
  const CurrentIcon = current.icon;

  useEffect(() => setOpen(false), [location.pathname]);
  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open]);

  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-[70] flex w-14 flex-col items-center border-r border-white/10 bg-[#07111f] text-white">
        <div className="grid h-15 w-full place-items-center border-b border-white/10">
          <span className="grid h-8.5 w-8.5 place-items-center rounded-[10px] bg-teal-400 text-[#07111f] shadow-lg shadow-teal-500/20">
            <Boxes size={18} strokeWidth={2.4} />
          </span>
        </div>

        <div className="flex flex-1 flex-col items-center gap-2 py-3">
          <button
            type="button"
            aria-label="Mở danh sách trang"
            aria-expanded={open}
            title="Mở điều hướng"
            className={`grid h-9 w-9 place-items-center rounded-[10px] transition ${
              open
                ? "bg-teal-400 text-[#07111f]"
                : "text-slate-300 hover:bg-white/10 hover:text-white"
            }`}
            onClick={() => setOpen((value) => !value)}
          >
            <Menu size={18} />
          </button>

          <div className="my-1 h-px w-7 bg-white/10" />

          <span
            className="grid h-9 w-9 place-items-center rounded-[10px] bg-white/8 text-teal-300"
            title={current.label}
            aria-label={`Trang hiện tại: ${current.label}`}
          >
            <CurrentIcon size={17} />
          </span>
        </div>

        <NavLink
          to="/settings"
          title="Settings"
          aria-label="Settings"
          className={({ isActive }) =>
            `mb-3 grid h-9 w-9 place-items-center rounded-[10px] transition ${
              isActive
                ? "bg-teal-400 text-[#07111f]"
                : "text-slate-400 hover:bg-white/10 hover:text-white"
            }`
          }
        >
          <Settings size={17} />
        </NavLink>
      </aside>

      {open && (
        <>
          <button
            type="button"
            aria-label="Đóng danh sách trang"
            className="fixed inset-y-0 left-14 right-0 z-50 cursor-default bg-slate-950/25 backdrop-blur-[2px]"
            onClick={() => setOpen(false)}
          />
          <aside
            role="dialog"
            aria-modal="true"
            aria-label="Danh sách trang"
            className="fixed bottom-2 left-16 top-2 z-[60] flex w-[286px] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl shadow-slate-950/20"
          >
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
              <div>
                <div className="text-sm font-bold text-slate-900">Điều hướng</div>
                <div className="text-[11px] text-slate-500">SimplyDent RAG Control Center</div>
              </div>
              <button
                type="button"
                aria-label="Đóng menu"
                className="grid h-8 w-8 place-items-center rounded-lg text-slate-500 hover:bg-slate-100"
                onClick={() => setOpen(false)}
              >
                <X size={16} />
              </button>
            </div>

            <nav className="flex-1 overflow-y-auto p-3">
              {navigationGroups.map((group) => (
                <div key={group.label} className="mb-4 last:mb-0">
                  <div className="mb-1.5 px-2 text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                    {group.label}
                  </div>
                  <div className="space-y-1">
                    {group.items.map(({ to, label, description, icon: Icon }) => (
                      <NavLink
                        key={to}
                        to={to}
                        className={({ isActive }) =>
                          `flex items-center gap-3 rounded-xl px-3 py-2.5 transition ${
                            isActive
                              ? "bg-teal-50 text-teal-800 ring-1 ring-teal-200"
                              : "text-slate-700 hover:bg-slate-50"
                          }`
                        }
                      >
                        {({ isActive }) => (
                          <>
                            <span
                              className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg ${
                                isActive
                                  ? "bg-teal-500 text-white"
                                  : "bg-slate-100 text-slate-500"
                              }`}
                            >
                              <Icon size={16} />
                            </span>
                            <span className="min-w-0">
                              <span className="block text-xs font-bold">{label}</span>
                              <span className="mt-0.5 block truncate text-[10px] text-slate-500">
                                {description}
                              </span>
                            </span>
                          </>
                        )}
                      </NavLink>
                    ))}
                  </div>
                </div>
              ))}
            </nav>
          </aside>
        </>
      )}
    </>
  );
}
