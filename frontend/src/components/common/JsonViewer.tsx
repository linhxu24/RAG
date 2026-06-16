export function JsonViewer({
  value,
  maxHeight = "360px",
}: {
  value: unknown;
  maxHeight?: string;
}) {
  return (
    <pre
      style={{ maxHeight }}
      className="mono overflow-auto rounded-xl border border-slate-800 bg-[#0b1524] p-4 text-xs leading-6 text-slate-200"
    >
      {JSON.stringify(value ?? {}, null, 2)}
    </pre>
  );
}
