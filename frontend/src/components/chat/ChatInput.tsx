import { SendHorizontal } from "lucide-react";

export function ChatInput({
  value,
  loading,
  onChange,
  onSend,
}: {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
}) {
  return (
    <div className="border-t border-slate-200 bg-white/95 p-4 backdrop-blur">
      <div className="mx-auto flex max-w-[900px] items-end gap-3 rounded-2xl border border-slate-200 bg-white p-2 shadow-[0_10px_35px_rgba(22,36,58,0.1)] focus-within:border-teal-400 focus-within:ring-4 focus-within:ring-teal-100">
        <textarea
          aria-label="Chat message"
          value={value}
          rows={1}
          placeholder="Hỏi về sản phẩm, dịch vụ hoặc thông tin phòng khám..."
          className="max-h-36 min-h-11 flex-1 resize-none border-0 bg-transparent px-3 py-2.5 text-sm leading-6 outline-none"
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
        />
        <button
          className="primary-button h-11 w-11 shrink-0 p-0"
          disabled={loading || !value.trim()}
          onClick={onSend}
          aria-label="Send message"
        >
          <SendHorizontal size={18} />
        </button>
      </div>
      <div className="mx-auto mt-2 max-w-[900px] text-center text-[10px] text-slate-400">
        Enter để gửi · Shift + Enter để xuống dòng
      </div>
    </div>
  );
}
