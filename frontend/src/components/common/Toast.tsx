import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { CheckCircle2, X, XCircle } from "lucide-react";

interface ToastMessage {
  id: number;
  message: string;
  type: "success" | "error";
}

const ToastContext = createContext<(message: string, type?: ToastMessage["type"]) => void>(
  () => undefined,
);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ToastMessage[]>([]);
  const notify = useCallback((message: string, type: ToastMessage["type"] = "success") => {
    const id = Date.now();
    setMessages((current) => [...current, { id, message, type }]);
    window.setTimeout(
      () => setMessages((current) => current.filter((item) => item.id !== id)),
      4200,
    );
  }, []);
  const value = useMemo(() => notify, [notify]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed right-6 top-20 z-[100] flex w-96 flex-col gap-2">
        {messages.map((item) => (
          <div
            key={item.id}
            className="flex items-start gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-xl"
          >
            {item.type === "success" ? (
              <CheckCircle2 className="mt-0.5 text-emerald-600" size={18} />
            ) : (
              <XCircle className="mt-0.5 text-rose-600" size={18} />
            )}
            <span className="flex-1 text-sm font-medium text-slate-700">
              {item.message}
            </span>
            <button
              className="text-slate-400 hover:text-slate-700"
              onClick={() =>
                setMessages((current) => current.filter((row) => row.id !== item.id))
              }
            >
              <X size={15} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export const useToast = () => useContext(ToastContext);
