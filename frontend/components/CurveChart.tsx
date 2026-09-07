"use client";

import type { FittedCurve } from "@/lib/api";

interface CurveChartProps {
  curve: FittedCurve;
}

const WIDTH = 640;
const HEIGHT = 320;
const PADDING = 36;

export default function CurveChart({ curve }: CurveChartProps) {
  const { x, y_observed, y_fitted } = curve;

  if (x.length === 0) {
    return null;
  }

  const allY = [...y_observed, ...y_fitted];
  const xMin = Math.min(...x);
  const xMax = Math.max(...x);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const scaleX = (v: number) =>
    PADDING + ((v - xMin) / xRange) * (WIDTH - 2 * PADDING);
  const scaleY = (v: number) =>
    HEIGHT - PADDING - ((v - yMin) / yRange) * (HEIGHT - 2 * PADDING);

  // Sort indices by x so the fitted line is drawn as a clean left-to-right path
  // even if the input observations weren't already sorted.
  const order = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);

  const fittedPath = order
    .map((i, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(x[i])} ${scaleY(y_fitted[i])}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="h-auto w-full rounded-lg bg-white"
      role="img"
      aria-label="Observed data points with the fitted curve overlaid"
    >
      <line
        x1={PADDING}
        y1={HEIGHT - PADDING}
        x2={WIDTH - PADDING}
        y2={HEIGHT - PADDING}
        stroke="#cbd5e1"
      />
      <line
        x1={PADDING}
        y1={PADDING}
        x2={PADDING}
        y2={HEIGHT - PADDING}
        stroke="#cbd5e1"
      />

      <path d={fittedPath} fill="none" stroke="#4f46e5" strokeWidth={2} />

      {x.map((xv, i) => (
        <circle
          key={i}
          cx={scaleX(xv)}
          cy={scaleY(y_observed[i])}
          r={3.5}
          fill="#0f172a"
          fillOpacity={0.55}
        />
      ))}

      <g transform={`translate(${WIDTH - PADDING - 150}, ${PADDING - 12})`}>
        <circle cx={0} cy={0} r={3.5} fill="#0f172a" fillOpacity={0.55} />
        <text x={10} y={4} fontSize={11} fill="#334155">
          observed
        </text>
        <line x1={70} y1={0} x2={85} y2={0} stroke="#4f46e5" strokeWidth={2} />
        <text x={90} y={4} fontSize={11} fill="#334155">
          fitted
        </text>
      </g>
    </svg>
  );
}
