import React, { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { TrendingUp, Sparkles, Search, Loader2, CheckCircle, AlertTriangle } from "lucide-react";

// Demo fallback data
const demoHistory = Array.from({ length: 90 }, (_, i) => {
  const base = 100;
  const noise = Math.sin(i / 7) * 2 + (Math.random() - 0.5);
  return { date: new Date(Date.now() - (90 - i) * 86400000).toISOString().slice(0, 10), close: +(base + i * 0.3 + noise).toFixed(2), volume: 500000 + Math.floor(Math.random() * 100000) };
});

const demoPredict = {
  best: { ticker: "AAPL", score: 0.74, cluster: 2 },
  alternatives: [
    { ticker: "MSFT", score: 0.68, cluster: 2 },
    { ticker: "NVDA", score: 0.65, cluster: 1 },
    { ticker: "AMZN", score: 0.63, cluster: 1 },
  ],
  regime: "Moderate volatility, consistent momentum"
};

const demoRecent = [
  { timestamp: new Date().toISOString(), ticker: "AAPL", score: 0.74, decision: "BUY" },
  { timestamp: new Date(Date.now() - 86400000).toISOString(), ticker: "MSFT", score: 0.68, decision: "CONSIDER" },
  { timestamp: new Date(Date.now() - 2 * 86400000).toISOString(), ticker: "TSLA", score: 0.52, decision: "HOLD" },
];

export default function StockAdvisorDashboard() {
  const [tickers, setTickers] = useState("AAPL, MSFT, NVDA, AMZN");
  const [range, setRange] = useState("6mo");
  const [selected, setSelected] = useState("AAPL");
  const [history, setHistory] = useState(demoHistory);
  const [predict, setPredict] = useState(demoPredict);
  const [recent, setRecent] = useState(demoRecent);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Get historical data
  const loadHistory = async (tkr = selected) => {
    try {
      setLoading(true); setError("");
      const res = await fetch(`/api/history?ticker=${tkr}&range=${range}`);
      if (!res.ok) throw new Error("history fetch failed");
      const data = await res.json();
      if (!Array.isArray(data) || data.length === 0) throw new Error("no history");
      setHistory(data);
    } catch {
      setHistory(demoHistory);
      setError("Showing demo data (history). Backend not reachable.");
    } finally { setLoading(false); }
  };

  // Run prediction
  const runPredict = async () => {
    try {
      setLoading(true); setError("");
      const body = { tickers: tickers.split(",").map(t => t.trim()).filter(Boolean), k: 3 };
      const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error("predict failed");
      const data = await res.json();
      setPredict(data);
      if (data?.best?.ticker) setSelected(data.best.ticker);
    } catch {
      setPredict(demoPredict);
      setSelected(demoPredict.best.ticker);
      setError("Showing demo data (prediction). Backend not reachable.");
    } finally { setLoading(false); }
  };

  // Load recent predictions
  const loadRecent = async () => {
    try {
      const res = await fetch("/api/recent-predictions");
      if (!res.ok) throw new Error("recent failed");
      const data = await res.json();
      setRecent(Array.isArray(data) && data.length ? data : demoRecent);
    } catch {
      setRecent(demoRecent);
    }
  };

  useEffect(() => {
    loadHistory(selected);
    loadRecent();
  }, []);

  useEffect(() => { if (selected) loadHistory(selected); }, [selected, range]);

  const best = predict?.best ?? {};
  const alts = predict?.alternatives ?? [];

  const kpiColor = useMemo(() => (best?.score ?? 0) >= 0.7 ? "text-green-600" : (best?.score ?? 0) >= 0.55 ? "text-yellow-700" : "text-red-600", [best]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-6 h-6" />
            <h1 className="text-xl font-semibold">Stock Advisor Dashboard</h1>
          </div>
          <div className="flex items-center gap-2">
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <CheckCircle className="w-5 h-5 text-emerald-600" />}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        {/* Search & Controls */}
        <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
          <div className="flex flex-col md:flex-row md:items-end gap-4">
            <div className="flex-1">
              <label className="block text-sm text-slate-600 mb-1">Tickers (comma-separated)</label>
              <input value={tickers} onChange={e => setTickers(e.target.value)} className="w-full rounded-xl border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-sky-300 outline-none" placeholder="AAPL, MSFT, NVDA" />
            </div>
            <div>
              <label className="block text-sm text-slate-600 mb-1">Range</label>
              <select value={range} onChange={e => setRange(e.target.value)} className="rounded-xl border border-slate-300 px-3 py-2 focus:ring-2 focus:ring-sky-300 outline-none">
                <option value="1mo">1 month</option>
                <option value="3mo">3 months</option>
                <option value="6mo">6 months</option>
                <option value="1y">1 year</option>
              </select>
            </div>
            <div className="flex gap-2">
              <button onClick={runPredict} className="inline-flex items-center gap-2 rounded-xl bg-slate-900 text-white px-4 py-2 hover:bg-slate-800">
                <Sparkles className="w-4 h-4" /> Recommend
              </button>
              <button onClick={() => loadHistory(selected)} className="inline-flex items-center gap-2 rounded-xl border border-slate-300 px-4 py-2 hover:bg-slate-100">
                <Search className="w-4 h-4" /> Refresh Chart
              </button>
            </div>
          </div>
          {error && (
            <div className="mt-3 inline-flex items-center gap-2 text-amber-700 bg-amber-50 border border-amber-200 rounded-xl px-3 py-2">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-sm">{error}</span>
            </div>
          )}
        </section>

        {/* KPI Section */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
            <div className="text-sm text-slate-500">Best Stock (Prescribed Choice)</div>
            <div className="mt-2 text-2xl font-semibold">{best?.ticker ?? "—"}</div>
            <div className={`mt-1 text-sm ${kpiColor}`}>Confidence Score: {(best?.score ?? 0).toFixed(2)}</div>
          </div>
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
            <div className="text-sm text-slate-500">Market Regime</div>
            <div className="mt-2 text-lg">{predict?.regime ?? "—"}</div>
          </div>
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
            <div className="text-sm text-slate-500">Currently Viewing</div>
            <div className="mt-2 text-2xl font-semibold">{selected}</div>
            <div className="mt-1 text-sm text-slate-600">Range: {range}</div>
          </div>
        </section>

        <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Price History – {selected}</h2>
          </div>

          {Array.isArray(history) && history.length > 0 ? (
            <div className="w-full overflow-x-auto">
              {/* Fixed-size chart to avoid ResponsiveContainer context */}
              <LineChart width={880} height={300} data={history} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" minTickGap={24} tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                <Tooltip formatter={(v) => [`$${Number(v).toFixed(2)}`, "Close"]} labelFormatter={(l) => `Date: ${l}`} />
                <Legend />
                <Line type="monotone" dataKey="close" strokeWidth={2} dot={false} name={`${selected} Close`} />
              </LineChart>
            </div>
          ) : (
            <div className="w-full h-72 flex items-center justify-center text-slate-500 text-sm">
              No data yet
            </div>
          )}
        </section>

        {/* Recent Predictions */}
        <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
          <h3 className="text-lg font-semibold mb-2">Recent Predictions</h3>
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b border-slate-200">
                  <th className="py-2 pr-4">Time</th>
                  <th className="py-2 pr-4">Ticker</th>
                  <th className="py-2 pr-4">Score</th>
                  <th className="py-2 pr-4">Decision</th>
                </tr>
              </thead>
              <tbody>
                {recent.map(({ timestamp, ticker, score, decision }, i) => (
                  <tr key={`${ticker}-${i}`} className="border-b border-slate-100">
                    <td className="py-2 pr-4">{new Date(timestamp).toLocaleString()}</td>
                    <td className="py-2 pr-4 font-medium">{ticker}</td>
                    <td className="py-2 pr-4">{score.toFixed(2)}</td>
                    <td className="py-2 pr-4">
                      <span className={`px-2 py-1 rounded-full text-xs ${decision === "BUY" ? "bg-emerald-50 text-emerald-700" : decision === "CONSIDER" ? "bg-amber-50 text-amber-700" : "bg-slate-100 text-slate-700"}`}>{decision}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <footer className="max-w-6xl mx-auto px-4 pb-10 text-xs text-slate-500">
        <p>This dashboard provides a prescriptive recommendation based on your trained K-Means model. Only one “best” stock is prescribed per run.</p>
      </footer>
    </div>
  );
}
