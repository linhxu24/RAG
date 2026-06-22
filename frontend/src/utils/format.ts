import { API_BASE_URL, PUBLIC_ASSETS_BASE_URL } from "../api/client";

export function formatPercent(value?: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value?: number | null, digits = 0): string {
  return value == null ? "—" : new Intl.NumberFormat("vi-VN", {
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatCurrency(
  value?: number | null,
  currency = "VND",
): string {
  if (value == null) return "—";
  try {
    return new Intl.NumberFormat("vi-VN", {
      style: "currency",
      currency: currency || "VND",
      maximumFractionDigits: currency === "VND" ? 0 : 2,
    }).format(value);
  } catch {
    return `${formatNumber(value)} ${currency}`.trim();
  }
}

export function formatDuration(minutes?: number | null): string {
  if (minutes == null) return "—";
  if (minutes < 60) return `${formatNumber(minutes)} phút`;
  const hours = Math.floor(minutes / 60);
  const remaining = minutes % 60;
  return remaining
    ? `${formatNumber(hours)} giờ ${formatNumber(remaining)} phút`
    : `${formatNumber(hours)} giờ`;
}

export function formatBoolean(value?: boolean | null): string {
  if (value == null) return "—";
  return value ? "Có" : "Không";
}

export function formatUnknownValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "boolean") return formatBoolean(value);
  if (Array.isArray(value)) return value.map(formatUnknownValue).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
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
