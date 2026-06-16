import type { ApiList, DocumentRecord } from "../types";
import { apiRequest } from "./client";

export const listDocuments = () =>
  apiRequest<ApiList<DocumentRecord>>("/api/documents");
export const getDocument = (id: string) =>
  apiRequest<DocumentRecord>(`/api/documents/${id}`);
export const approveDocument = (id: string) =>
  apiRequest(`/api/documents/${id}/approve`, { method: "POST" });
export const activateDocument = (id: string) =>
  apiRequest(`/api/documents/${id}/activate`, { method: "POST" });
export const archiveDocument = (id: string) =>
  apiRequest(`/api/documents/${id}/archive`, { method: "POST" });
export const reingestDocument = (id: string) =>
  apiRequest(`/api/documents/${id}/reingest`, {
    method: "POST",
    timeoutMs: 130_000,
  });
export const deleteDocument = (id: string) =>
  apiRequest(`/api/documents/${id}?confirm=true`, { method: "DELETE" });
export const resetApplicationData = (
  scope: "content" | "runtime",
  confirmation: string,
) =>
  apiRequest<{
    scope: string;
    deleted: Record<string, number>;
    remaining: Record<string, number>;
  }>("/api/admin/data-reset", {
    method: "POST",
    body: JSON.stringify({ scope, confirmation }),
  });

export interface UploadOptions {
  documentType: string;
  extractTables: boolean;
  extractAssets: boolean;
  createEmbeddings: boolean;
  requireReview: boolean;
  duplicatePolicy?: "reject" | "reuse" | "replace" | "force";
}

export async function uploadDocument(
  file: File,
  options: UploadOptions,
  assetFiles: File[] = [],
) {
  const form = new FormData();
  form.append("file", file);
  form.append("document_type", options.documentType);
  form.append("extract_tables", String(options.extractTables));
  form.append("extract_assets", String(options.extractAssets));
  form.append("create_embeddings", String(options.createEmbeddings));
  form.append("require_review", String(options.requireReview));
  form.append("duplicate_policy", options.duplicatePolicy ?? "reject");
  assetFiles.forEach((asset) => form.append("asset_files", asset));
  return apiRequest<{
    doc_id: string;
    run_id: string;
    document_status: string;
    run_status: string;
    detected_document_type?: string | null;
    document_type_confidence?: number | null;
    quality_report?: Record<string, any>;
  }>("/ingest/upload", { method: "POST", body: form, timeoutMs: 180_000 });
}
