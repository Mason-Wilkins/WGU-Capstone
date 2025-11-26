// src/App.js
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
  BarChart, Bar, ScatterChart, Scatter
} from "recharts";
import {
  TrendingUp, Sparkles, Search, Loader2, CheckCircle, AlertTriangle
} from "lucide-react";
import "./App.css";

// Normalize API base (strip trailing slashes). Default to local Flask dev server.
const RAW_API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:5000";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");
const API_KEY = process.env.REACT_APP_API_KEY || "";
const AUTH_HEADERS = API_KEY ? { "X-API-Key": API_KEY } : {};
const RANGE_LIMITS = { d: 365 * 3, mo: 36, y: 3 };
const LINE_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#0d9488",
  "#f97316", "#0891b2", "#f43f5e", "#7c3aed", "#84cc16"
];

/* ================================
   Feature engineering / wrangling
   ================================ */

function engineer(history) {
  if (!Array.isArray(history) || history.length === 0) {
    return {
      enriched: [],
      returnsSeries: [],
      cumSeries: [],
      volSeries: [],
      momSeries: [],
      mvScatter: [],
      hasEnough: false,
    };
  }

  const toNum = (x) => (typeof x === "number" ? x : Number(x));
  const enriched = history.map((d) => ({
    date: d.date,
    close: toNum(d.close),
    volume: toNum(d.volume),
  }));

  // simple moving average helper
  const sma = (arr, w, key) => {
    const out = new Array(arr.length).fill(null);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i][key];
      if (i >= w) sum -= arr[i - w][key];
      if (i >= w - 1) out[i] = sum / w;
    }
    return out;
  };

  // MA5 and MA20 (short- and medium-term)
  const ma5 = sma(enriched, 5, "close");
  const ma20 = sma(enriched, 20, "close");

  // daily returns
  const returns = new Array(enriched.length).fill(null);
  for (let i = 1; i < enriched.length; i++) {
    const prev = enriched[i - 1].close;
    const cur = enriched[i].close;
    returns[i] = prev ? (cur - prev) / prev : null;
  }

  // rolling mean and std (20-day)
  const rolling = (arr, w) => {
    const mean = new Array(arr.length).fill(null);
    const stdev = new Array(arr.length).fill(null);
    const buf = [];
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      buf.push(typeof v === "number" && isFinite(v) ? v : null);
      if (buf.length > w) buf.shift();
      if (buf.length === w && buf.every((x) => typeof x === "number")) {
        const m = buf.reduce((a, b) => a + b, 0) / w;
        const sd = Math.sqrt(buf.reduce((a, b) => a + Math.pow(b - m, 2), 0) / w);
        mean[i] = m;
        stdev[i] = sd;
      }
    }
    return { mean, stdev };
  };
  const { mean: mom20, stdev: vol20 } = rolling(returns, 20);

  // cumulative return from first valid
  const cumSeries = [];
  let base = enriched.find((d) => typeof d.close === "number")?.close ?? null;
  for (let i = 0; i < enriched.length; i++) {
    const c = enriched[i].close;
    cumSeries.push({
      date: enriched[i].date,
      cum: base && c ? c / base - 1 : null,
    });
  }

  const enrichedWithMA = enriched.map((d, i) => ({
    ...d,
    ma5: ma5[i],
    ma20: ma20[i],
  }));

  const returnsSeries = enriched.map((d, i) => ({ date: d.date, ret: returns[i] }));
  const volSeries = enriched.map((d, i) => ({ date: d.date, vol20: vol20[i] }));
  const momSeries = enriched.map((d, i) => ({ date: d.date, mom20: mom20[i] }));

  // scatter: momentum vs volatility
  const mvScatter = [];
  for (let i = 0; i < enriched.length; i++) {
    if (typeof vol20[i] === "number" && typeof mom20[i] === "number") {
      mvScatter.push({ vol: vol20[i], mom: mom20[i], date: enriched[i].date });
    }
  }

  return {
    enriched: enrichedWithMA,
    returnsSeries,
    cumSeries,
    volSeries,
    momSeries,
    mvScatter,
    hasEnough: mvScatter.length > 0,
  };
}

function buildHistogram(returnsSeries, bins = 20) {
  const vals = (returnsSeries || [])
    .map((d) => d.ret)
    .filter((x) => typeof x === "number" && isFinite(x));
  if (vals.length === 0) return { bars: [], domain: [0, 0] };

  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const width = (max - min) / bins || 1e-6;

  const buckets = Array.from({ length: bins }, (_, i) => ({
    x0: min + i * width,
    x1: min + (i + 1) * width,
    count: 0,
  }));

  for (const v of vals) {
    let idx = Math.floor((v - min) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    buckets[idx].count += 1;
  }

  const bars = buckets.map((b) => ({
    bin: `${(b.x0 * 100).toFixed(2)}% – ${(b.x1 * 100).toFixed(2)}%`,
    mid: (b.x0 + b.x1) / 2,
    count: b.count,
  }));

  return { bars, domain: [min, max] };
}

/* ================================
   Main App
   ================================ */

export default function StockAdvisorDashboard() {
  // text input (for ad-hoc symbol)
  const [tickers, setTickers] = useState("");
  // selected bubbles
  const [selectedTickers, setSelectedTickers] = useState([]);

  const [rangeValue, setRangeValue] = useState(6);
  const [rangeUnit, setRangeUnit] = useState("mo"); // "d" | "mo" | "y"
  const [selected, setSelected] = useState("");
  const [historyMap, setHistoryMap] = useState({});     // {symbol: [{date, close, volume}]}
  const [predict, setPredict] = useState(null);   // { best, alternatives, regime }
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // chart options
  const [riskProfile, setRiskProfile] = useState("balanced");
  const [momentumWindow, setMomentumWindow] = useState(20);
  const [volWindow, setVolWindow] = useState(20);

  // API health
  const [apiHealth, setApiHealth] = useState({ ok: false, ms: null });

  // suggestions
  const [suggest, setSuggest] = useState([]);
  const [showSuggest, setShowSuggest] = useState(false);
  const [suggestLoading, setSuggestLoading] = useState(false);
  const [sampleClusters, setSampleClusters] = useState({ points: [], centroids: [], silhouette: null });

  const handleClearAll = useCallback(() => {
    setSelectedTickers([]);
    setTickers("");
    setPredict(null);
    setSelected("");
    setHistoryMap({});
    setError("");
    setShowSuggest(false);
    setSuggest([]);
  }, []);

  const normalizeRangeValue = useCallback((value, unit) => {
    const limit = RANGE_LIMITS[unit] ?? RANGE_LIMITS.mo;
    const safeVal = Number.isFinite(value) ? value : 1;
    return Math.min(limit, Math.max(1, Math.trunc(safeVal)));
  }, []);

  const rangeParam = useMemo(() => `${rangeValue}${rangeUnit}`, [rangeValue, rangeUnit]);
  const rangeLabel = useMemo(() => {
    const unitLabel = rangeUnit === "d" ? "day" : rangeUnit === "mo" ? "month" : "year";
    const plural = rangeValue === 1 ? "" : "s";
    return `${rangeValue} ${unitLabel}${plural}`;
  }, [rangeValue, rangeUnit]);

  /* ----- Load history for chart ----- */
  const loadHistory = useCallback(
    async (tkr, { silent = false } = {}) => {
      if (!tkr) return;
      const ticker = tkr.toUpperCase();
      const t0 = performance.now();
      try {
        if (!silent) {
          setLoading(true);
          setError("");
        }
        const qs = new URLSearchParams({ ticker, range: rangeParam }).toString();
        const res = await fetch(`${API_BASE}/api/history?${qs}`, {
          headers: AUTH_HEADERS,
          credentials: "include",
        });
        const t1 = performance.now();
        if (!silent) setApiHealth({ ok: res.ok, ms: Math.round(t1 - t0) });

        if (!res.ok) throw new Error(`History fetch failed: ${res.status}`);
        const data = await res.json();
        if (!Array.isArray(data) || data.length === 0) {
          if (!silent) setError(`No history returned for ${ticker}.`);
          setHistoryMap((prev) => ({ ...prev, [ticker]: [] }));
          return;
        }
        setHistoryMap((prev) => ({ ...prev, [ticker]: data }));
      } catch (e) {
        if (!silent) {
          const msg = e?.message || "Failed to load history.";
          const maybeCORS = msg.includes("Failed to fetch") || msg.includes("TypeError");
          setError(
            maybeCORS
              ? `${msg} — Is Flask running on ${API_BASE}? CORS configured for http://localhost:3000?`
              : msg
          );
          setApiHealth({ ok: false, ms: null });
        }
        setHistoryMap((prev) => ({ ...prev, [ticker]: [] }));
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [rangeParam]
  );

  /* ----- Run prescriptive endpoint ----- */
  const runPredict = async () => {
    // Merge bubbles + free-text, uppercase & de-dup
    const list = [...selectedTickers];
    if (tickers.trim()) list.push(tickers.trim().toUpperCase());
    const uniqueList = Array.from(new Set(list)).filter(Boolean);

    if (uniqueList.length === 0) {
      setError("Select or type at least one ticker.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const body = {
        tickers: uniqueList,
        k: 3,
        range: rangeParam,
        // NEW knobs wired to the backend:
        momentumWindow,
        volWindow,
        riskProfile,      // "conservative" | "balanced" | "aggressive"
      };

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...AUTH_HEADERS },
        credentials: "include",
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
      const data = await res.json();
      setPredict(data || null);

      const bestTicker = data?.best?.ticker || "";
      if (bestTicker) {
        setSelected(bestTicker);
        await loadHistory(bestTicker);
        const others = uniqueList.filter((sym) => sym !== bestTicker);
        await Promise.all(others.map((sym) => loadHistory(sym, { silent: true })));
      } else {
        setSelected("");
        const skipped = data?.debug?.skipped;
        if (skipped && Object.keys(skipped).length) {
          const reasons = Object.entries(skipped)
            .map(([sym, reason]) => `${sym}: ${reason}`)
            .join("; ");
          setError(`Insufficient usable data. ${reasons}`);
        } else {
          setError(data?.regime || "No best recommendation returned by API.");
        }
      }
      await loadRecent();
    } catch (e) {
      const msg = e?.message || "Failed to run prediction.";
      const maybeCORS = msg.includes("Failed to fetch") || msg.includes("TypeError");
      setPredict(null);
      setSelected("");
      setError(maybeCORS ? `${msg} — Is Flask running on ${API_BASE}? CORS allows http://localhost:3000?` : msg);
    } finally {
      setLoading(false);
    }
  };


  /* ----- Recent predictions (monitoring) ----- */
  const loadRecent = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/recent-predictions`, {
        headers: AUTH_HEADERS,
        credentials: "include",
      });
      if (!res.ok) return; // silently ignore
      const data = await res.json();
      if (Array.isArray(data)) setRecent(data);
    } catch {/* ignore */}
  }, []);

  useEffect(() => {
    loadRecent();
    const id = setInterval(loadRecent, 30000);
    return () => clearInterval(id);
  }, [loadRecent]);

  useEffect(() => {
    const fetchSampleClusters = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/clusters/sample?limit=25`, {
          headers: AUTH_HEADERS,
          credentials: "include",
        });
        if (!res.ok) return;
        const data = await res.json();
        setSampleClusters({
          points: Array.isArray(data?.points) ? data.points : [],
          centroids: Array.isArray(data?.centroids) ? data.centroids : [],
          silhouette: Number.isFinite(data?.silhouette) ? data.silhouette : null,
        });
      } catch {
        setSampleClusters({ points: [], centroids: [], silhouette: null });
      }
    };
    fetchSampleClusters();
  }, []);

  useEffect(() => {
    if (selected) loadHistory(selected);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, loadHistory]);

  useEffect(() => {
    const loadAll = async () => {
      await Promise.all(
        selectedTickers.map((sym) => loadHistory(sym, { silent: true }))
      );
    };
    if (selectedTickers.length) {
      loadAll();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTickers, rangeParam, loadHistory]);

  /* ----- Suggestions (debounced) ----- */
  useEffect(() => {
    const q = tickers.trim();
    if (!q) {
      setSuggest([]);
      setShowSuggest(false);
      return;
    }
    const id = setTimeout(async () => {
      try {
        setSuggestLoading(true);
        const qs = new URLSearchParams({ q, limit: 16 }).toString();
        const res = await fetch(`${API_BASE}/api/tickers/suggest?${qs}`, {
          headers: AUTH_HEADERS,
          credentials: "include",
        });
        if (!res.ok) throw new Error("suggest failed");
        const data = await res.json();
        // Expecting: [{symbol, name}] from your server
        setSuggest(Array.isArray(data) ? data : []);
        setShowSuggest(true);
      } catch {
        setSuggest([]);
        setShowSuggest(false);
      } finally {
        setSuggestLoading(false);
      }
    }, 180);
    return () => clearTimeout(id);
  }, [tickers]);

  /* ----- Derived data for charts ----- */
  const selectedHistory = useMemo(
    () => historyMap[selected] || [],
    [historyMap, selected]
  );
  const {
    enriched,
    returnsSeries,
    cumSeries,
    volSeries,
    mvScatter,
    hasEnough,
  } = useMemo(() => engineer(selectedHistory), [selectedHistory]);

  const multiHistory = useMemo(() => {
    const rowsByDate = new Map();
    const activeSymbols = new Set();

    Object.entries(historyMap).forEach(([symbol, rows]) => {
      if (!rows || rows.length === 0) return;
      activeSymbols.add(symbol);
      rows.forEach(({ date, close }) => {
        const key = date;
        if (!rowsByDate.has(key)) rowsByDate.set(key, { date: key });
        rowsByDate.get(key)[symbol] = close;
      });
    });

    const merged = Array.from(rowsByDate.values()).sort(
      (a, b) => new Date(a.date) - new Date(b.date)
    );
    return { data: merged, symbols: Array.from(activeSymbols) };
  }, [historyMap]);

  const multiMetrics = useMemo(() => {
    const volumeByDate = new Map();
    const returnsByDate = new Map();
    const volByDate = new Map();
    const cumByDate = new Map();
    const symbols = Object.keys(historyMap || {}).filter(
      (k) => Array.isArray(historyMap[k]) && historyMap[k].length > 0
    );

    const std = (arr) => {
      const m = arr.reduce((a, b) => a + b, 0) / arr.length;
      const v = arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
      return Math.sqrt(v);
    };

    const upsert = (map, date, ticker, value) => {
      const key = date;
      if (!map.has(key)) map.set(key, { date: key });
      map.get(key)[ticker] = value;
    };

    symbols.forEach((ticker) => {
      const rows = (historyMap[ticker] || []).slice().sort((a, b) => new Date(a.date) - new Date(b.date));
      if (!rows.length) return;
      let prevClose = null;
      let base = null;
      const retWindow = [];
      rows.forEach((row) => {
        const close = Number(row.close);
        const vol = Number(row.volume);
        if (base === null && Number.isFinite(close)) base = close;

        // Volume
        if (Number.isFinite(vol)) upsert(volumeByDate, row.date, ticker, vol);

        // Returns
        let ret = null;
        if (Number.isFinite(close) && Number.isFinite(prevClose) && prevClose !== 0) {
          ret = (close - prevClose) / prevClose;
        }
        if (ret !== null) {
          upsert(returnsByDate, row.date, ticker, ret);
          retWindow.push(ret);
          if (retWindow.length > 20) retWindow.shift();
          if (retWindow.length === 20) {
            upsert(volByDate, row.date, ticker, std(retWindow));
          }
        }
        prevClose = Number.isFinite(close) ? close : prevClose;

        // Cumulative
        if (Number.isFinite(close) && Number.isFinite(base) && base !== 0) {
          upsert(cumByDate, row.date, ticker, close / base - 1);
        }
      });
    });

    const toSorted = (map) =>
      Array.from(map.values()).sort((a, b) => new Date(a.date) - new Date(b.date));

    return {
      symbols,
      volumeData: toSorted(volumeByDate),
      returnsData: toSorted(returnsByDate),
      volData: toSorted(volByDate),
      cumData: toSorted(cumByDate),
    };
  }, [historyMap]);


  const { bars: histBars } = useMemo(
    () => buildHistogram(returnsSeries, 20),
    [returnsSeries]
  );

  const clusterSeries = useMemo(() => {
    const feats = predict?.raw_features || {};
    const labels = predict?.labels || {};
    const buckets = {};
    Object.entries(feats).forEach(([sym, vals]) => {
      const cluster = Number.isFinite(labels[sym]) ? Number(labels[sym]) : null;
      if (cluster === null) return;
      const vol = Number(vals?.volatility);
      const mom = Number(vals?.momentum);
      if (!Number.isFinite(vol) || !Number.isFinite(mom)) return;
      if (!buckets[cluster]) buckets[cluster] = [];
      buckets[cluster].push({ ticker: sym, vol, mom, cluster });
    });
    return Object.entries(buckets)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([cluster, points]) => ({ cluster: Number(cluster), points }));
  }, [predict]);

  const clusterCenters = useMemo(() => {
    return clusterSeries.map(({ cluster, points }) => {
      const vol = points.reduce((s, p) => s + p.vol, 0) / points.length;
      const mom = points.reduce((s, p) => s + p.mom, 0) / points.length;
      return { cluster, vol, mom };
    });
  }, [clusterSeries]);

  const clusterTickerLabels = useMemo(() => {
    const out = {};
    clusterSeries.forEach(({ cluster, points }) => {
      const names = points.map((p) => p.ticker).sort();
      const head = names.slice(0, 6).join(", ");
      const more = names.length > 6 ? `, +${names.length - 6} more` : "";
      out[cluster] = `${head}${more}`;
    });
    return out;
  }, [clusterSeries]);

  const skippedTickers = useMemo(() => predict?.skipped || {}, [predict]);

  const silhouetteScore = useMemo(() => {
    const s = predict?.silhouette;
    if (Number.isFinite(s)) return s;
    if (Number.isFinite(sampleClusters?.silhouette)) return sampleClusters.silhouette;
    return null;
  }, [predict, sampleClusters]);

  const recentScores = useMemo(
    () =>
      (recent || []).slice().reverse().map((r, idx) => ({
        idx,
        time: new Date(r.timestamp).toLocaleString(),
        score: r.score,
      })),
    [recent]
  );

  const best = useMemo(() => (predict?.best ? predict.best : null), [predict]);
  const kpiColor = useMemo(() => {
    const s = best?.score ?? 0;
    if (s >= 0.7) return "kpi-green";
    if (s >= 0.55) return "kpi-amber";
    return "kpi-red";
  }, [best]);

  /* ================================
     UI
     ================================ */

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            <TrendingUp className="icon" />
            <h1 className="brand-title">Stock Advisor Dashboard</h1>
          </div>
          <div className="status">
            <span className="api-health">
              API: {apiHealth.ok ? "OK" : "—"} {apiHealth.ms ? `(${apiHealth.ms}ms)` : ""}
            </span>
            {loading ? <Loader2 className="spin icon" /> : <CheckCircle className="ok icon" />}
          </div>
        </div>
      </header>

      <main className="container">
        {/* Controls */}
        <section className="card">
          <div className="controls">
            <div className="field grow" style={{ position: "relative" }}>
              <label className="label">Tickers</label>

              {/* Selected bubbles + Clear All */}
              <div className="bubble-row">
                <div className="bubble-container">
                  {selectedTickers.map((sym) => (
                    <div key={sym} className="ticker-bubble" title={sym}>
                      {sym}
                      <button
                        className="bubble-remove"
                        onClick={() => setSelectedTickers(selectedTickers.filter((s) => s !== sym))}
                        aria-label={`Remove ${sym}`}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  type="button"
                  className="btn subtle"
                  onClick={handleClearAll}
                  disabled={selectedTickers.length === 0 && !tickers.trim()}
                  title="Clear all selected tickers and input"
                >
                  Clear All
                </button>
              </div>

              {/* Text input for ad-hoc symbol/company search */}
              <input
                value={tickers}
                onChange={(e) => setTickers(e.target.value.toUpperCase())}
                onBlur={() => setTimeout(() => setShowSuggest(false), 150)}
                onFocus={() => setShowSuggest(suggest.length > 0)}
                className="input"
                placeholder="Type a ticker or company name (e.g., AAPL, MICROSOFT)"
                autoComplete="off"
              />

              {/* Suggestions as bubbles */}
              {showSuggest && (
                <div className="suggest-popover">
                  {suggestLoading && <div className="suggest-item muted">Searching…</div>}
                  {!suggestLoading && suggest.length === 0 && (
                    <div className="suggest-item muted">No matches</div>
                  )}
                  {!suggestLoading &&
                    suggest.map(({ symbol, name }) => (
                      <button
                        key={symbol}
                        type="button"
                        className="suggest-item"
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => {
                          const sym = String(symbol || "").toUpperCase();
                          if (sym && !selectedTickers.includes(sym)) {
                            setSelectedTickers([...selectedTickers, sym]);
                          }
                          setTickers("");
                          setShowSuggest(false);
                        }}
                        title={name}
                      >
                        <span className="suggest-symbol">{symbol.toUpperCase()}</span>
                        <span className="suggest-name">{name || "\u00A0"}</span>
                      </button>
                    ))}
                </div>
              )}
            </div>

            <div className="field">
              <label className="label">Range</label>
              <div className="range-control">
                <input
                  type="number"
                  min={1}
                  max={RANGE_LIMITS[rangeUnit]}
                  value={rangeValue}
                  onChange={(e) =>
                    setRangeValue(
                      normalizeRangeValue(Number(e.target.value) || 1, rangeUnit)
                    )
                  }
                  className="input"
                />
                <select
                  value={rangeUnit}
                  onChange={(e) => {
                    const nextUnit = e.target.value;
                    setRangeUnit(nextUnit);
                    setRangeValue((val) => normalizeRangeValue(val, nextUnit));
                  }}
                  className="select"
                >
                  <option value="d">Days</option>
                  <option value="mo">Months</option>
                  <option value="y">Years</option>
                </select>
              </div>
            </div>

            <div className="btn-row">
              <button onClick={runPredict} className="btn primary">
                <Sparkles className="icon-sm" /> Recommend
              </button>
              <button
                onClick={() => selected && loadHistory(selected)}
                className="btn"
                disabled={!selected}
              >
                <Search className="icon-sm" /> Refresh Chart
              </button>
            </div>
          </div>

         <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="field">
            <label className="label">Risk Profile</label>
            <select
              value={riskProfile}
              onChange={e => setRiskProfile(e.target.value)}
              className="select"
            >
              <option value="conservative">Conservative</option>
              <option value="balanced">Balanced</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>

          <div className="field">
            <label className="label">Momentum Window (days)</label>
            <input
              type="number"
              min={5}
              max={90}
              value={momentumWindow}
              onChange={e => setMomentumWindow(Number(e.target.value))}
              className="input"
            />
          </div>

          <div className="field">
            <label className="label">Volatility Window (days)</label>
            <input
              type="number"
              min={5}
              max={60}
              value={volWindow}
              onChange={e => setVolWindow(Number(e.target.value))}
              className="input"
            />
          </div>

        </div>
          {error && (
            <div className="alert">
              <AlertTriangle className="icon-sm" />
              <span>{error}</span>
            </div>
          )}
          {!predict && !error && (
            <p className="muted">
              Select bubbles or type symbols, then click <strong>Recommend</strong>.
            </p>
          )}

        </section>

        {/* KPI Row */}
        <section className="grid kpi-grid kpi-grid-wide">
          <div className="card kpi">
            <div className="kpi-title">Best Stock (Prescribed Choice)</div>
            <div className="kpi-main">{best?.ticker ?? "—"}</div>
            <div className={`kpi-sub ${kpiColor}`}>
              Confidence Score: {(best?.score ?? 0).toFixed(2)}
            </div>
          </div>
          <div className="card kpi">
            <div className="kpi-title">Market Snapshot</div>
            <div className="kpi-main small">{predict?.regime ?? "—"}</div>
          </div>
          <div className="card kpi">
            <div className="kpi-title">Currently Viewing</div>
            <div className="kpi-main">{selected || "—"}</div>
            <div className="kpi-sub">Range: {rangeLabel}</div>
          </div>
        </section>

        {/* 1) Price History (multi-line) */}
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Price History ({multiHistory.symbols.length || "0"} tickers)</h2>
          </div>
          <p className="muted">Overlay of closing prices for all selected tickers over the chosen range to compare trends side by side.</p>
          {multiHistory.data.length > 0 ? (
            <div className="chart-wrap">
              <LineChart width={880} height={300} data={multiHistory.data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" minTickGap={24} />
                <YAxis domain={["auto", "auto"]} />
                <Tooltip />
                <Legend />
                {multiHistory.symbols.map((sym, idx) => (
                  <Line
                    key={sym}
                    type="monotone"
                    dataKey={sym}
                    strokeWidth={sym === selected ? 3 : 2}
                    dot={false}
                    name={sym}
                    stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                    opacity={sym === selected ? 1 : 0.7}
                  />
                ))}
              </LineChart>
            </div>
          ) : (
            <div className="empty">{selected ? "No history to display." : "No ticker selected."}</div>
          )}
        </section>

        {/* 2) Volume (Bar) */}
        <section className="card">
          <h3 className="card-title mb">Daily Volume {selected ? `– ${selected}` : ""}</h3>
          <p className="muted">Shows trading activity; thin volume can make prices noisy and is filtered out by the model.</p>
          {enriched.length > 0 ? (
            <div className="chart-wrap">
              <BarChart width={880} height={240} data={enriched} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" minTickGap={24} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="volume" name="Volume" />
              </BarChart>
            </div>
          ) : (
            <div className="empty">No data.</div>
          )}
        </section>

        {/* 3) Returns Histogram */}
        <section className="card">
          <h3 className="card-title mb">Daily Returns Histogram</h3>
          <p className="muted">Distribution of day-to-day returns; wider spread means higher volatility.</p>
          {histBars.length > 0 ? (
            <div className="chart-wrap">
              <BarChart
                width={880}
                height={240}
                data={histBars}
                margin={{ top: 30, right: 10, bottom: 60, left: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" angle={-35} textAnchor="end" interval={0} tickMargin={12} />
                <YAxis />
                <Tooltip />
                <Legend verticalAlign="top" align="left" />
                <Bar dataKey="count" name="Count" />
              </BarChart>
            </div>
          ) : (
            <div className="empty">No return data.</div>
          )}
        </section>

        {/* 4) Rolling Volatility */}
        <section className="card">
          <h3 className="card-title mb">Rolling Volatility (20D Std of Returns)</h3>
          <p className="muted">Tracks how turbulent the recent price moves are; the model prefers lower volatility for conservative risk.</p>
          {volSeries.length > 0 ? (
            <div className="chart-wrap">
              <LineChart width={880} height={220} data={volSeries} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" minTickGap={24} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="vol20" strokeWidth={2} dot={false} name="Volatility 20D" />
              </LineChart>
            </div>
          ) : (
            <div className="empty">No volatility data.</div>
          )}
        </section>

        {/* 5) Cumulative Returns */}
        <section className="card">
          <h3 className="card-title mb">Cumulative Return (from start of range)</h3>
          <p className="muted">Shows the total return from the first date in the selected range; good for seeing overall trend direction.</p>
          {cumSeries.length > 0 ? (
            <div className="chart-wrap">
              <LineChart width={880} height={220} data={cumSeries} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" minTickGap={24} />
                <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => [`${(v * 100).toFixed(2)}%`, "Cumulative"]} />
                <Legend />
                <Line type="monotone" dataKey="cum" strokeWidth={2} dot={false} name="Cumulative Return" />
              </LineChart>
            </div>
          ) : (
            <div className="empty">No cumulative data.</div>
          )}
        </section>

        {/* 6) Recent Prediction Scores */}
        {recentScores.length > 0 && (
          <section className="card">
            <h3 className="card-title mb">Recent Prediction Scores</h3>
            <p className="muted">Recent confidence scores logged by the model; higher means stronger BUY/CONSIDER leaning at the time.</p>
            <div className="chart-wrap">
              <LineChart width={880} height={220} data={recentScores} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" minTickGap={24} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="score" strokeWidth={2} dot name="Score" />
              </LineChart>
            </div>
          </section>
        )}

        {/* 7) Momentum vs Volatility (Scatter) */}
        <section className="card">
          <h3 className="card-title mb">Momentum vs Volatility (20D) {selected ? `– ${selected}` : ""}</h3>
          <p className="muted">Rolling momentum vs. volatility for the selected ticker; used as inputs to clustering and scoring.</p>
          {hasEnough ? (
            <div className="chart-wrap">
              <ScatterChart width={880} height={260} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="vol" name="Volatility 20D" />
                <YAxis type="number" dataKey="mom" name="Momentum 20D" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <Scatter name="Points" data={mvScatter} />
              </ScatterChart>
            </div>
          ) : (
            <div className="empty">Not enough rolling-window points yet.</div>
          )}
        </section>

        {/* 8) Clustering view (current prediction run) */}
        <section className="card">
          <h3 className="card-title mb">Clustering (volatility vs. momentum)</h3>
          <p className="muted">
            Shows how K-Means groups tickers by volatility and momentum. Colors = your latest /api/predict run; diamonds = centroids (cluster centers).
            The gray “reference cloud” is a baseline clustering of common tickers so you can compare your run against a broader sample.
          </p>
          {Number.isFinite(silhouetteScore) && (
            <div className="kpi-row">
              <div className="kpi">
                <div className="kpi-label">Silhouette (cluster quality)</div>
                <div className="kpi-main">{silhouetteScore.toFixed(3)}</div>
                <div className="kpi-sub">
                  Scale [-1, 1]: near 1 means points are tight and well-separated; near 0 means clusters overlap; below 0 means likely mis-clustered.
                </div>
              </div>
            </div>
          )}
          {clusterSeries.length > 0 || (sampleClusters.points && sampleClusters.points.length > 0) ? (
            <div className="chart-wrap">
              <ScatterChart width={880} height={260} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="vol" name="Volatility" tickFormatter={(v) => v.toFixed(3)} />
                <YAxis type="number" dataKey="mom" name="Momentum" tickFormatter={(v) => v.toFixed(3)} />
                <Tooltip formatter={(v, name, props) => [Number(v).toFixed(4), name]} labelFormatter={() => ""} />
                <Legend />
                {sampleClusters.points?.length > 0 && (
                  <Scatter
                    name="Reference cloud"
                    data={sampleClusters.points}
                    fill="#cbd5e1"
                    opacity={0.6}
                  />
                )}
                {clusterSeries.map(({ cluster, points }) => (
                  <Scatter
                    key={cluster}
                    name={`Cluster ${cluster}`}
                    data={points}
                    fill={LINE_COLORS[cluster % LINE_COLORS.length]}
                  />
                ))}
                {sampleClusters.centroids?.length > 0 && (
                  <Scatter
                    name="Reference centroids"
                    data={sampleClusters.centroids}
                    shape="diamond"
                    legendType="diamond"
                    fill="#94a3b8"
                  />
                )}
                {clusterCenters.map(({ cluster, vol, mom }) => (
                  <Scatter
                    key={`c-${cluster}`}
                    name={`Centroid ${cluster}${clusterTickerLabels[cluster] ? ` (${clusterTickerLabels[cluster]})` : ""}`}
                    data={[{ vol, mom }]}
                    shape="diamond"
                    legendType="diamond"
                    fill={LINE_COLORS[cluster % LINE_COLORS.length]}
                  />
                ))}
              </ScatterChart>
            </div>
          ) : (
            <div className="empty">No clustering results yet. Run a prediction to populate.</div>
          )}
          {clusterSeries.length > 0 && (
            <div className="muted" style={{ marginTop: "8px" }}>
              {clusterSeries.map(({ cluster }) => (
                <div key={`cluster-label-${cluster}`}>
                  <strong>Cluster {cluster}:</strong> {clusterTickerLabels[cluster] || "—"}
                </div>
              ))}
            </div>
          )}
          {skippedTickers && Object.keys(skippedTickers).length > 0 && (
            <div className="muted" style={{ marginTop: "8px" }}>
              <strong>Filtered out:</strong>{" "}
              {Object.entries(skippedTickers)
                .map(([sym, reason]) => `${sym}: ${reason}`)
                .join("; ")}
            </div>
          )}
        </section>

        {/* Recent Predictions */}
        {recent.length > 0 && (
          <section className="card">
            <h3 className="card-title mb">Recent Predictions</h3>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Ticker</th>
                    <th>Score</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map(({ timestamp, ticker, score, decision }, i) => (
                    <tr key={`${ticker}-${i}`}>
                      <td>{new Date(timestamp).toLocaleString()}</td>
                      <td className="bold">{ticker}</td>
                      <td>{Number(score).toFixed(2)}</td>
                      <td>
                        <span
                          className={`pill ${
                            decision === "BUY"
                              ? "pill-green"
                              : decision === "CONSIDER"
                              ? "pill-amber"
                              : "pill-gray"
                          }`}
                        >
                          {decision}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>
          All insights are for educational purposes only. Always perform your own research and consult
          a qualified financial advisor; the developers assume no liability for trading losses.
        </p>
      </footer>
    </div>
  );
}
