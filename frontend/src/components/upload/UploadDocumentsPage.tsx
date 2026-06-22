import { useMutation } from "@tanstack/react-query";
import {
  CheckCircle2,
  FileUp,
  Image,
  Layers3,
  ScanText,
  UploadCloud,
} from "lucide-react";
import { useRef, useState } from "react";

import { uploadDocument, type UploadOptions } from "../../api/documents";
import { useToast } from "../common/Toast";
import { PageContainer } from "../layout/PageContainer";

const documentTypes = [
  ["auto", "Auto Detect"],
  ["product", "Product Document"],
  ["service", "Service Document"],
  ["faq", "FAQ Document"],
  ["clinic_info", "Clinic Info"],
  ["policy", "Policy"],
  ["unknown", "Unknown"],
];

export function UploadDocumentsPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const notify = useToast();
  const [file, setFile] = useState<File>();
  const [assetFiles, setAssetFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [options, setOptions] = useState<UploadOptions>({
    documentType: "auto",
    extractTables: true,
    extractAssets: true,
    createEmbeddings: true,
    requireReview: false,
    duplicatePolicy: "reject",
  });
  const mutation = useMutation({
    mutationFn: () => {
      if (!file) throw new Error("Vui lòng chọn file.");
      return uploadDocument(file, options, assetFiles);
    },
    onSuccess: (data) =>
      notify(
        data.document_status === "active"
          ? "Ingestion hoàn tất và document đã active."
          : `Ingestion hoàn tất với trạng thái ${data.document_status}. Hãy kiểm tra quality report.`,
        data.document_status === "active" ? "success" : "error",
      ),
    onError: (error) => notify(error.message, "error"),
  });

  const choose = (selected?: File) => {
    if (selected) setFile(selected);
  };

  return (
    <PageContainer
      title="Upload Documents"
      description="Upload tài liệu vào Docling ingestion pipeline và kiểm soát cách xử lý dữ liệu."
    >
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.3fr)_minmax(360px,0.7fr)]">
        <div className="space-y-5">
          <div
            className={`panel flex min-h-[360px] flex-col items-center justify-center border-2 border-dashed p-10 text-center transition ${
              dragging
                ? "border-teal-400 bg-teal-50"
                : "border-slate-300 bg-white hover:border-teal-300"
            }`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              choose(event.dataTransfer.files[0]);
            }}
          >
            <span className="grid h-16 w-16 place-items-center rounded-2xl bg-teal-50 text-teal-700">
              <UploadCloud size={29} />
            </span>
            <h2 className="mt-5 text-lg font-bold text-slate-900">
              Kéo thả tài liệu vào đây
            </h2>
            <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
              Hỗ trợ PDF, DOCX, TXT, CSV, XLSX và ảnh. Ảnh được lưu bằng Asset Masking,
              không caption bằng Vision LLM mặc định.
            </p>
            <button
              className="secondary-button mt-5 px-4 py-2.5 text-sm"
              onClick={() => inputRef.current?.click()}
            >
              <FileUp size={17} /> Chọn file
            </button>
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              onChange={(event) => choose(event.target.files?.[0])}
            />
            {file && (
              <div className="mt-6 flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3">
                <CheckCircle2 size={18} className="text-emerald-600" />
                <div className="text-left">
                  <div className="text-sm font-bold text-emerald-800">{file.name}</div>
                  <div className="text-xs text-emerald-600">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                </div>
              </div>
            )}
          </div>
          {mutation.data && (
            <div className="panel border-emerald-200 bg-emerald-50/50 p-5">
              <div className="flex items-center gap-2 font-bold text-emerald-800">
                <CheckCircle2 size={18} /> Upload result
              </div>
              <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
                {[
                  ["doc_id", mutation.data.doc_id],
                  ["ingestion_run_id", mutation.data.run_id],
                  ["document_status", mutation.data.document_status],
                  ["run_status", mutation.data.run_status],
                  [
                    "detected_document_type",
                    mutation.data.detected_document_type || "unknown",
                  ],
                  [
                    "document_type_confidence",
                    mutation.data.document_type_confidence?.toFixed(2) || "—",
                  ],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-xl bg-white p-3">
                    <div className="font-bold uppercase tracking-wider text-slate-400">
                      {label}
                    </div>
                    <div className="mono mt-1 break-all text-slate-700">{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <aside className="panel h-fit p-5">
          <h2 className="font-bold text-slate-900">Ingestion options</h2>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Backend hiện ingest đầy đủ pipeline; metadata options được gửi kèm để tương thích
            với API mở rộng.
          </p>
          <label className="mt-5 block text-xs font-bold text-slate-600">
            Document type
          </label>
          <select
            className="control mt-2 px-3 py-2.5 text-sm"
            value={options.documentType}
            onChange={(event) =>
              setOptions({ ...options, documentType: event.target.value })
            }
          >
            {documentTypes.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <div className="mt-5 space-y-3">
            <Option
              icon={Layers3}
              label="Extract tables"
              checked={options.extractTables}
              onChange={(checked) => setOptions({ ...options, extractTables: checked })}
            />
            <Option
              icon={Image}
              label="Extract assets / images"
              checked={options.extractAssets}
              onChange={(checked) => setOptions({ ...options, extractAssets: checked })}
            />
            <Option
              icon={ScanText}
              label="Create embeddings"
              checked={options.createEmbeddings}
              onChange={(checked) =>
                setOptions({ ...options, createEmbeddings: checked })
              }
            />
            <Option
              icon={CheckCircle2}
              label="Require human review"
              checked={options.requireReview}
              onChange={(checked) => setOptions({ ...options, requireReview: checked })}
            />
          </div>
          <label className="mt-5 block text-xs font-bold text-slate-600">
            Duplicate policy
          </label>
          <select
            className="control mt-2 px-3 py-2.5 text-sm"
            value={options.duplicatePolicy}
            onChange={(event) =>
              setOptions({
                ...options,
                duplicatePolicy: event.target.value as UploadOptions["duplicatePolicy"],
              })
            }
          >
            <option value="reject">Reject identical content</option>
            <option value="reuse">Reuse existing document</option>
            <option value="replace">Replace active version</option>
            <option value="force">Create another version</option>
          </select>
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Dùng Replace khi cập nhật catalog đã ingest. Reject sẽ trả lỗi 409 nếu checksum
            đã tồn tại.
          </p>
          <label className="mt-5 block text-xs font-bold text-slate-600">
            Companion product/service images
          </label>
          <input
            type="file"
            accept="image/*"
            multiple
            className="control mt-2 px-3 py-2 text-xs"
            onChange={(event) => setAssetFiles(Array.from(event.target.files || []))}
          />
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Tên file phải khớp cột <span className="mono">image_reference</span> trong
            CSV/XLSX, ví dụ <span className="mono">oral_b_pro_500.png</span>.
          </p>
          {assetFiles.length > 0 && (
            <div className="mt-2 rounded-xl bg-slate-50 p-3 text-xs text-slate-600">
              {assetFiles.length} ảnh: {assetFiles.map((item) => item.name).join(", ")}
            </div>
          )}
          <button
            className="primary-button mt-6 w-full px-4 py-3 text-sm"
            disabled={!file || mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            <UploadCloud size={17} />
            {mutation.isPending ? "Đang ingest..." : "Upload & Run Ingestion"}
          </button>
        </aside>
      </div>
    </PageContainer>
  );
}

function Option({
  icon: Icon,
  label,
  checked,
  onChange,
}: {
  icon: typeof Layers3;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-slate-200 p-3 hover:bg-slate-50">
      <span className="rounded-lg bg-slate-100 p-2 text-slate-600">
        <Icon size={16} />
      </span>
      <span className="flex-1 text-sm font-semibold text-slate-700">{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-4 w-4 accent-teal-600"
      />
    </label>
  );
}
