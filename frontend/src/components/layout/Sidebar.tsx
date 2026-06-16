import {
  Activity,
  Bot,
  Boxes,
  Database,
  FileSearch,
  Gauge,
  Image,
  SearchCode,
  Settings,
  SquareStack,
  UploadCloud,
} from "lucide-react";
import { NavLink } from "react-router-dom";

const links = [
  { to: "/chatbot", label: "Chatbot", icon: Bot },
  { to: "/upload", label: "Upload Documents", icon: UploadCloud },
  { to: "/documents", label: "Document Store", icon: SquareStack },
  { to: "/ingestion", label: "Ingestion Monitor", icon: Activity },
  { to: "/retrieval", label: "Retrieval Playground", icon: SearchCode },
  { to: "/evaluation", label: "Evaluation Dashboard", icon: Gauge },
  { to: "/observability", label: "Observability", icon: Activity },
  { to: "/traces", label: "Trace Explorer", icon: FileSearch },
  { to: "/assets", label: "Asset Manager", icon: Image },
  { to: "/data", label: "Data Tables", icon: Database },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 z-40 flex w-[252px] flex-col border-r border-white/10 bg-[#07111f] text-white">
      <div className="flex h-[72px] items-center gap-3 border-b border-white/10 px-5">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-teal-400 text-[#07111f] shadow-lg shadow-teal-500/20">
          <Boxes size={22} strokeWidth={2.3} />
        </span>
        <div>
          <div className="font-bold tracking-tight">Dental RAG</div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
            Control Center
          </div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-5">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-[13px] font-semibold transition ${
                isActive
                  ? "bg-teal-400 text-[#07111f] shadow-lg shadow-teal-950/20"
                  : "text-slate-300 hover:bg-white/7 hover:text-white"
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-white/10 p-4">
        <div className="rounded-xl bg-white/5 p-3">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Workspace
          </div>
          <div className="mt-1 text-xs font-semibold text-slate-300">SimplyDent RAG</div>
        </div>
      </div>
    </aside>
  );
}
