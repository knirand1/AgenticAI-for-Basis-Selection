export type BasisParams = Record<string, number>;

export interface LeaderboardEntry {
  name: string;
  gcv: number;
  n_basis: number;
  params: BasisParams;
}

export interface FittedCurve {
  x: number[];
  y_observed: number[];
  y_fitted: number[];
}

export interface AnalyzeResponse {
  selected_basis: string;
  selected_params: BasisParams;
  gcv: number;
  rationale: string;
  leaderboard: LeaderboardEntry[];
  fitted_curve: FittedCurve;
}

export interface AnalyzeRequest {
  x: number[];
  y: number[];
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL;

export class AnalyzeError extends Error {}

/**
 * Extracts a human-readable message from a FastAPI error response body.
 * A plain HTTPException produces {"detail": "some string"}; a Pydantic
 * validation error (422) produces {"detail": [{"msg": "...", "loc": [...]}, ...]}.
 */
function extractErrorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in (body as Record<string, unknown>)) {
    const detail = (body as { detail: unknown }).detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            const loc = Array.isArray((item as { loc?: unknown }).loc)
              ? (item as { loc: unknown[] }).loc.join(".")
              : undefined;
            const msg = String((item as { msg: unknown }).msg);
            return loc ? `${loc}: ${msg}` : msg;
          }
          return String(item);
        })
        .join("; ");
    }
  }
  return fallback;
}

export async function analyze(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new AnalyzeError(
      `Could not reach the backend at ${API_BASE_URL}. Is it running?`
    );
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // No JSON body (or empty response) - fall through to status-based message.
  }

  if (!response.ok) {
    throw new AnalyzeError(
      extractErrorMessage(body, `Request failed with status ${response.status}.`)
    );
  }

  return body as AnalyzeResponse;
}
