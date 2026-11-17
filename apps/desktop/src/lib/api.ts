import { createApiClient } from "@finsight/api-client";

export const apiClient = createApiClient({
  baseUrl: import.meta.env.VITE_API_URL || "http://localhost:8000",
});
