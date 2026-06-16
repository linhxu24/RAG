import { API_BASE_URL, PUBLIC_ASSETS_BASE_URL } from "../api/client";

export function formatPercent(value?: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value?: number | null, digits = 0): string {
  return value == null ? "—" : new Intl.NumberFormat("vi-VN", {
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatLatency(value?: number | null): string {
  return value == null ? "—" : `${formatNumber(value)} ms`;
}

export function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat("vi-VN", {
        dateStyle: "short",
        timeStyle: "medium",
      }).format(date);
}

export function truncate(value?: string | null, length = 46): string {
  if (!value) return "—";
  return value.length > length ? `${value.slice(0, length)}…` : value;
}

export function assetUrl(url?: string | null): string | null {
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/assets/")) return `${API_BASE_URL}${url}`;
  if (url.startsWith("/")) return `${API_BASE_URL}${url}`;
  return `${PUBLIC_ASSETS_BASE_URL}/${url.replace(/^\/+/, "")}`;
}
