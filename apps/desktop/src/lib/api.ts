import { createApiClient } from "@finsight/api-client";

const API_URL_STORAGE_KEY = "finsight_api_url";

function normalizeBaseUrl(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

export function getStoredApiBaseUrl(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const val = window.localStorage.getItem(API_URL_STORAGE_KEY);
    if (!val) {
      return null;
    }
    const normalized = normalizeBaseUrl(val);
    return normalized.length > 0 ? normalized : null;
  } catch {
    return null;
  }
}

export function setStoredApiBaseUrl(rawUrl: string): void {
  if (typeof window === "undefined") {
    return;
  }
  const normalized = normalizeBaseUrl(rawUrl);
  if (normalized.length === 0) {
    window.localStorage.removeItem(API_URL_STORAGE_KEY);
  } else {
    window.localStorage.setItem(API_URL_STORAGE_KEY, normalized);
  }
}

export function resolveApiBaseUrl(): string {
  const stored = getStoredApiBaseUrl();
  if (stored) {
    return stored;
  }
  if (import.meta.env.VITE_API_URL) {
    return normalizeBaseUrl(import.meta.env.VITE_API_URL);
  }
  if (
    typeof window !== "undefined" &&
    window.location.hostname.endsWith(".app.github.dev")
  ) {
    // Route via Vite dev proxy on the same origin inside GitHub Codespaces to avoid cross-port CORS/auth issues.
    return window.location.origin;
  }
  return "http://localhost:8000";
}

export const CURRENT_API_BASE_URL = resolveApiBaseUrl();

export const apiClient = createApiClient({
  baseUrl: CURRENT_API_BASE_URL,
});
