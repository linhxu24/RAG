import { useMutation, useQuery } from "@tanstack/react-query";
import { Bot, ChevronDown, HelpCircle, Search, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { sendChat } from "../../api/chat";
import { listPublicFaqs } from "../../api/faqs";
import { getTrace } from "../../api/traces";
import type { ChatMessage, ChatResponse, TraceRecord } from "../../types";
import { AssistantMessageCard } from "./AssistantMessageCard";
import { ChatInput } from "./ChatInput";
import { DebugPanel } from "./DebugPanel";

const CHAT_SESSION_STORAGE_KEY = "simplydent.chat.session_id";

function getOrCreateSessionId() {
  try {
    const existing = window.localStorage.getItem(CHAT_SESSION_STORAGE_KEY);
    if (existing) return existing;
    const created = crypto.randomUUID();
    window.localStorage.setItem(CHAT_SESSION_STORAGE_KEY, created);
    return created;
  } catch {
    return crypto.randomUUID();
  }
}

const sessionId = getOrCreateSessionId();

export function ChatPage() {
  const [activeTab, setActiveTab] = useState<"chat" | "faq">("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [value, setValue] = useState("");
  const [latestResponse, setLatestResponse] = useState<ChatResponse>();
  const [trace, setTrace] = useState<TraceRecord>();
  const scrollRef = useRef<HTMLDivElement>(null);
  const mutation = useMutation({
    mutationFn: (message: string) => sendChat(message, sessionId),
  });

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, mutation.isPending]);

  const send = async () => {
    const text = value.trim();
    if (!text || mutation.isPending) return;
    setValue("");
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text,
      createdAt: new Date().toISOString(),
    };
    setMessages((current) => [...current, userMessage]);
    try {
      const response = await mutation.mutateAsync(text);
      setLatestResponse(response);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: response.answer?.text || response.message?.text || "",
          response,
          createdAt: new Date().toISOString(),
        },
      ]);
      if (response.trace_id) {
        try {
          setTrace(await getTrace(response.trace_id));
        } catch {
          setTrace(undefined);
        }
      }
    } catch (error) {
      const apiError = error as ApiError;
      const traceId =
        apiError.traceId ||
        apiError.message.match(/trace_id=([0-9a-f-]+)/i)?.[1] ||
        "unavailable";
      const response: ChatResponse = {
        trace_id: traceId,
        error: {
          type: apiError.code || "CHAT_REQUEST_FAILED",
          message: apiError.message,
          failed_step: apiError.failedStep,
        },
      };
      setLatestResponse(response);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: apiError.message,
          response,
          error: apiError.message,
          createdAt: new Date().toISOString(),
        },
      ]);
      if (traceId !== "unavailable") {
        try {
          setTrace(await getTrace(traceId));
        } catch {
          setTrace(undefined);
        }
      }
    }
  };

  const latest = useMemo(
    () => [...messages].reverse().find((item) => item.role === "assistant"),
    [messages],
  );

  return (
    <div className="h-[calc(100vh-72px)] overflow-hidden">
      <div className="flex h-[73px] items-center justify-between border-b border-slate-200 bg-white px-6">
        <div className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-teal-50 text-teal-700">
            <Bot size={20} />
          </span>
          <div>
            <h1 className="font-bold text-slate-900">Dental Chatbot</h1>
            <p className="text-xs text-slate-400">
              Truy vấn dữ liệu có cấu trúc, FAQ và tài liệu đã ingest.
            </p>
          </div>
        </div>
        <div className="flex rounded-xl bg-slate-100 p-1">
          {(["chat", "faq"] as const).map((tab) => (
            <button
              key={tab}
              className={`rounded-lg px-4 py-2 text-xs font-bold uppercase ${
                activeTab === tab
                  ? "bg-white text-teal-700 shadow-sm"
                  : "text-slate-500"
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>
      {activeTab === "faq" ? (
        <FaqPanel />
      ) : (
      <div className="grid h-[calc(100%-73px)] grid-cols-1 xl:grid-cols-[minmax(0,1fr)_330px]">
        <section className="flex h-full min-h-0 min-w-0 flex-col bg-[#f7f9fc]">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
            {!messages.length ? (
              <Welcome onExample={setValue} />
            ) : (
              <div className="space-y-6">
                {messages.map((message) =>
                  message.role === "user" ? (
                    <div key={message.id} className="flex justify-end">
                      <div className="max-w-[68%] rounded-2xl rounded-br-md bg-[#10253f] px-4 py-3 text-sm leading-6 text-white shadow-md">
                        {message.text}
                      </div>
                    </div>
                  ) : (
                    <AssistantMessageCard key={message.id} message={message} />
                  ),
                )}
                {mutation.isPending && (
                  <div className="mx-auto flex max-w-[760px] items-center gap-3 rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-500">
                    <Sparkles size={17} className="text-teal-600" />
                    Đang xử lý
                    <span className="typing-dot">●</span>
                    <span className="typing-dot [animation-delay:150ms]">●</span>
                    <span className="typing-dot [animation-delay:300ms]">●</span>
                  </div>
                )}
              </div>
            )}
          </div>
          <ChatInput
            value={value}
            loading={mutation.isPending}
            onChange={setValue}
            onSend={() => void send()}
          />
        </section>
        <div className="hidden min-h-0 xl:block">
          <DebugPanel
            response={latest?.response || latestResponse}
            trace={trace}
            loading={mutation.isPending}
          />
        </div>
      </div>
      )}
    </div>
  );
}

function FaqPanel() {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");
  const categoryQuery = useQuery({
    queryKey: ["public-faq-categories"],
    queryFn: () => listPublicFaqs(),
    staleTime: 5 * 60_000,
  });
  const query = useQuery({
    queryKey: ["public-faqs", search, category],
    queryFn: () => listPublicFaqs(search, category),
  });
  const categories = Array.from(
    new Map(
      (categoryQuery.data?.items || [])
        .filter((item) => item.category_code)
        .map((item) => [item.category_code!, item.category || item.category_code!]),
    ),
  ).sort((left, right) => String(left[1]).localeCompare(String(right[1]), "vi"));
  return (
    <div className="h-[calc(100%-73px)] overflow-y-auto bg-[#f7f9fc] px-6 py-6">
      <div className="mx-auto max-w-4xl">
        <div className="panel p-5">
          <div className="flex items-center gap-3">
            <HelpCircle className="text-teal-700" size={22} />
            <div>
              <h2 className="font-bold text-slate-900">Câu hỏi thường gặp</h2>
              <p className="text-xs text-slate-500">
                Câu trả lời được đọc trực tiếp từ bảng FAQ đang active.
              </p>
            </div>
          </div>
          <div className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-[1fr_220px]">
            <label className="control flex items-center gap-2 px-3">
              <Search size={16} className="text-slate-400" />
              <input
                className="w-full bg-transparent py-2.5 text-sm outline-none"
                placeholder="Tìm câu hỏi..."
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
            <select
              className="control px-3 text-sm"
              value={category}
              onChange={(event) => setCategory(event.target.value)}
            >
              <option value="">Tất cả danh mục</option>
              {categories.map(([code, label]) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="mt-4 space-y-3">
          {query.isLoading ? (
            <div className="panel p-6 text-sm text-slate-500">Đang tải FAQ...</div>
          ) : query.isError ? (
            <div className="panel p-6 text-sm text-red-600">
              Không thể tải danh sách FAQ.
            </div>
          ) : !query.data?.items.length ? (
            <div className="panel p-6 text-sm text-slate-500">
              Không tìm thấy câu hỏi phù hợp.
            </div>
          ) : (
            query.data.items.map((faq) => (
              <details key={faq.faq_id} className="panel group overflow-hidden">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 text-sm font-bold text-slate-800">
                  {faq.question}
                  <ChevronDown
                    size={17}
                    className="shrink-0 transition group-open:rotate-180"
                  />
                </summary>
                <div className="border-t border-slate-100 px-5 py-4 text-sm leading-6 text-slate-600">
                  {faq.answer}
                </div>
              </details>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function Welcome({ onExample }: { onExample: (value: string) => void }) {
  const examples = [
    "Cho tôi danh sách sản phẩm đang có",
    "Dịch vụ tẩy trắng răng giá bao nhiêu?",
    "Địa chỉ và giờ làm việc của phòng khám?",
  ];
  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center pb-24 text-center">
      <span className="grid h-16 w-16 place-items-center rounded-2xl bg-[#10253f] text-teal-300 shadow-xl shadow-slate-300">
        <Sparkles size={27} />
      </span>
      <h2 className="mt-5 text-2xl font-bold tracking-tight text-slate-900">
        Kiểm thử chatbot nha khoa
      </h2>
      <p className="mt-2 max-w-xl text-sm leading-6 text-slate-500">
        Câu trả lời được grounded từ PostgreSQL, pgvector và tài liệu đã ingest.
        Debug panel sẽ hiển thị trace của từng request.
      </p>
      <div className="mt-7 grid w-full grid-cols-1 gap-3 md:grid-cols-3">
        {examples.map((example) => (
          <button
            key={example}
            className="rounded-2xl border border-slate-200 bg-white p-4 text-left text-xs font-semibold leading-5 text-slate-600 shadow-sm hover:border-teal-300 hover:text-teal-800"
            onClick={() => onExample(example)}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
