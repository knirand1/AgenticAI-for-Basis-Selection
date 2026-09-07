"use client";

import { useState, type FormEvent } from "react";
import { analyze, AnalyzeError, type AnalyzeResponse } from "@/lib/api";
import CurveChart from "@/components/CurveChart";
import Leaderboard from "@/components/Leaderboard";

type Status = "idle" | "loading" | "success" | "error";

const SAMPLE_X =
  "0.0, 0.345, 0.69, 1.034, 1.379, 1.724, 2.069, 2.414, 2.759, 3.103, 3.448, 3.793, 4.138, 4.483, 4.828, 5.172, 5.517, 5.862, 6.207, 6.552, 6.897, 7.241, 7.586, 7.931, 8.276, 8.621, 8.966, 9.31, 9.655, 10.0";
const SAMPLE_Y =
  "0.075, 0.317, 0.733, 1.088, 0.947, 0.953, 1.115, 0.78, 0.303, 0.12, -0.371, -0.676, -0.803, -1.261, -1.252, -0.98, -0.845, -0.362, -0.212, 0.053, 0.795, 0.784, 0.974, 0.783, 0.831, 0.737, 0.271, 0.171, -0.318, -0.588";

function parseNumberList(input: string): number[] | null {
  const parts = input
    .split(/[\s,]+/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  if (parts.length === 0) {
    return null;
  }

  const numbers = parts.map(Number);
  if (numbers.some((n) => Number.isNaN(n))) {
    return null;
  }

  return numbers;
}

export default function Page() {
  const [xInput, setXInput] = useState(SAMPLE_X);
  const [yInput, setYInput] = useState(SAMPLE_Y);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const loadSample = () => {
    setXInput(SAMPLE_X);
    setYInput(SAMPLE_Y);
    setValidationError(null);
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setValidationError(null);

    const x = parseNumberList(xInput);
    const y = parseNumberList(yInput);

    if (!x || !y) {
      setValidationError("x and y must be comma- or whitespace-separated numbers.");
      return;
    }
    if (x.length !== y.length) {
      setValidationError(
        `x and y must have the same length (got ${x.length} and ${y.length}).`
      );
      return;
    }
    if (x.length < 5) {
      setValidationError("Provide at least 5 observations (20+ recommended for a stable fit).");
      return;
    }

    setStatus("loading");
    setErrorMessage(null);

    try {
      const data = await analyze({ x, y });
      setResult(data);
      setStatus("success");
    } catch (err) {
      setErrorMessage(
        err instanceof AnalyzeError || err instanceof Error
          ? err.message
          : "Something went wrong."
      );
      setStatus("error");
    }
  };

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold text-slate-900">Basis-Selection Agent</h1>
        <p className="mt-1 text-sm text-slate-500">
          Submit (x, y) observations to the{" "}
          <code className="rounded bg-slate-100 px-1 py-0.5">/analyze</code> endpoint and see
          which basis family fits best, with a plain-English rationale.
        </p>
      </header>

      <form
        onSubmit={handleSubmit}
        className="mb-8 space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <div>
          <label htmlFor="x-input" className="mb-1 block text-sm font-medium text-slate-700">
            x values
          </label>
          <textarea
            id="x-input"
            value={xInput}
            onChange={(e) => setXInput(e.target.value)}
            rows={3}
            className="w-full rounded-lg border border-slate-300 p-2 font-mono text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            placeholder="e.g. 0, 0.5, 1, 1.5, 2, ..."
          />
        </div>

        <div>
          <label htmlFor="y-input" className="mb-1 block text-sm font-medium text-slate-700">
            y values
          </label>
          <textarea
            id="y-input"
            value={yInput}
            onChange={(e) => setYInput(e.target.value)}
            rows={3}
            className="w-full rounded-lg border border-slate-300 p-2 font-mono text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            placeholder="e.g. 0.1, 0.4, 0.9, 0.8, ..."
          />
        </div>

        {validationError && <p className="text-sm text-red-600">{validationError}</p>}

        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={status === "loading"}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {status === "loading" && (
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
            )}
            {status === "loading" ? "Running..." : "Run agent"}
          </button>
          <button
            type="button"
            onClick={loadSample}
            className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            Load sample data
          </button>
        </div>
      </form>

      {status === "error" && errorMessage && (
        <div className="mb-8 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <p className="font-medium">Request failed</p>
          <p className="mt-1">{errorMessage}</p>
        </div>
      )}

      {status === "success" && result && (
        <section className="space-y-6">
          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-lg font-semibold text-slate-900">
                Selected basis: <span className="text-indigo-600">{result.selected_basis}</span>
              </h2>
              <span className="font-mono text-sm text-slate-500">
                GCV = {result.gcv.toFixed(5)}
              </span>
            </div>
            <div className="mt-3 space-y-1 text-sm text-slate-600">
              {result.rationale.split("\n").map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <h3 className="mb-3 text-sm font-medium text-slate-700">Fitted curve</h3>
            <CurveChart curve={result.fitted_curve} />
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <h3 className="mb-3 text-sm font-medium text-slate-700">Leaderboard</h3>
            <Leaderboard entries={result.leaderboard} selected={result.selected_basis} />
          </div>
        </section>
      )}
    </main>
  );
}
