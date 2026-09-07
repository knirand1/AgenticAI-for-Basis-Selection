import type { LeaderboardEntry } from "@/lib/api";

function formatParams(params: Record<string, number>): string {
  return Object.entries(params)
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
}

export default function Leaderboard({
  entries,
  selected,
}: {
  entries: LeaderboardEntry[];
  selected: string;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full text-sm">
        <thead className="bg-slate-100 text-left text-slate-600">
          <tr>
            <th className="px-4 py-2 font-medium">Basis family</th>
            <th className="px-4 py-2 font-medium">GCV</th>
            <th className="px-4 py-2 font-medium"># basis fns</th>
            <th className="px-4 py-2 font-medium">Params</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr
              key={entry.name}
              className={
                entry.name === selected
                  ? "bg-indigo-50 font-medium text-indigo-900"
                  : "text-slate-700"
              }
            >
              <td className="px-4 py-2">
                {entry.name}
                {entry.name === selected && (
                  <span className="ml-2 rounded-full bg-indigo-600 px-2 py-0.5 text-xs text-white">
                    selected
                  </span>
                )}
              </td>
              <td className="px-4 py-2 font-mono">{entry.gcv.toFixed(5)}</td>
              <td className="px-4 py-2">{entry.n_basis}</td>
              <td className="px-4 py-2 font-mono text-xs">
                {formatParams(entry.params)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
