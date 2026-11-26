import { createApiClient } from "@finsight/api-client";

function resolveApiBaseUrl(): string {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
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

export const apiClient = createApiClient({
  baseUrl: resolveApiBaseUrl(),
});
