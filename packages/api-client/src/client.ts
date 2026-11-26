import createClient, { type Client } from "openapi-fetch";
import type { paths } from "./schema";

export interface ApiErrorDetail {
  field?: string;
  message: string;
  code?: string;
}

export interface ApiProblemDetails {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  instance?: string;
  code?: string;
  request_id?: string;
  errors?: ApiErrorDetail[];
}

export class ApiError extends Error {
  public readonly status: number;
  public readonly code?: string;
  public readonly details: ApiProblemDetails;

  constructor(status: number, details: ApiProblemDetails) {
    super(details.detail || details.title || `API Error ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.code = details.code;
    this.details = details;
  }
}

export interface ClientConfig {
  baseUrl?: string;
  getAccessToken?: () => string | null | Promise<string | null>;
  onUnauthorized?: () => void;
}

export function createApiClient(config: ClientConfig = {}): Client<paths> {
  const baseUrl = config.baseUrl ?? "http://localhost:8000";

  const client = createClient<paths>({
    baseUrl,
  });

  client.use({
    async onRequest({ request }) {
      if (config.getAccessToken) {
        const token = await config.getAccessToken();
        if (token) {
          request.headers.set("Authorization", `Bearer ${token}`);
        }
      }
      return request;
    },
    async onResponse({ response }) {
      if (response.status === 401 && config.onUnauthorized) {
        config.onUnauthorized();
      }
      if (!response.ok) {
        let problem: ApiProblemDetails = {
          status: response.status,
          title: response.statusText,
        };
        try {
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("json")) {
            problem = await response.clone().json();
          }
        } catch {
          // ignore json parse error
        }
        throw new ApiError(response.status, problem);
      }
      return response;
    },
  });

  return client;
}
