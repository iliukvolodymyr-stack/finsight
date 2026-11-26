import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, type components } from "@finsight/api-client";
import { apiClient } from "@/lib/api";
import { formatDateTime } from "@finsight/shared";
import {
  ColorType,
  LineSeries,
  createChart,
  type LineData,
  type UTCTimestamp,
} from "lightweight-charts";
import { Activity, AlertCircle, BarChart3, CheckCircle2, Cpu, Download, Play } from "lucide-react";

type EquityPoint = components["schemas"]["EquityPoint"];
type BacktestMetricDict = Record<string, number | string | boolean | null | undefined>;

function downloadTradesCsv(
  btName: string,
  items: components["schemas"]["BacktestTradeItem"][]
): void {
  const header = ["seq", "side", "entry_at", "entry_price", "exit_at", "exit_price", "quantity", "fee", "pnl", "exit_reason", "confidence", "fold_no"];
  const rows = items.map((t) =>
    [
      t.seq,
      t.side,
      t.entry_at,
      t.entry_price,
      t.exit_at,
      t.exit_price,
      t.quantity,
      t.fee,
      t.pnl,
      t.exit_reason,
      t.confidence ?? "",
      t.fold_no ?? "",
    ].join(",")
  );
  const blob = new Blob([[header.join(","), ...rows].join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${btName.toLowerCase().replace(/[^a-z0-9]+/g, "_")}_trades.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function EquityCurveChart({ items }: { items: EquityPoint[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || items.length === 0) {
      return;
    }

    const chart = createChart(el, {
      width: el.clientWidth,
      height: 300,
      layout: {
        background: { type: ColorType.Solid, color: "#020617" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "rgba(30, 41, 59, 0.45)" },
        horzLines: { color: "rgba(30, 41, 59, 0.45)" },
      },
      rightPriceScale: { borderColor: "#1e293b" },
      timeScale: { borderColor: "#1e293b", timeVisible: true },
    });

    const stratSeries = chart.addSeries(LineSeries, {
      color: "#10b981",
      lineWidth: 2,
      title: "LGBM Strategy",
    });
    const bhSeries = chart.addSeries(LineSeries, {
      color: "#6366f1",
      lineWidth: 2,
      title: "Buy & Hold",
    });
    const emaSeries = chart.addSeries(LineSeries, {
      color: "#f59e0b",
      lineWidth: 1,
      title: "EMA 20/50",
    });

    const toData = (fn: (p: EquityPoint) => number | null | undefined): LineData<UTCTimestamp>[] =>
      items
        .map((p) => ({ time: Math.floor(Date.parse(p.t) / 1000) as UTCTimestamp, val: fn(p) }))
        .filter((d): d is { time: UTCTimestamp; val: number } => typeof d.val === "number")
        .map((d) => ({ time: d.time, value: d.val }));

    stratSeries.setData(toData((p) => p.strategy));
    bhSeries.setData(toData((p) => p.benchmark));
    emaSeries.setData(toData((p) => p.ema_crossover));
    chart.timeScale().fitContent();

    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect && rect.width > 0) {
        chart.applyOptions({ width: Math.floor(rect.width) });
      }
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [items]);

  if (items.length === 0) {
    return (
      <div className="h-[300px] rounded-lg border border-slate-800 bg-slate-950/60 flex items-center justify-center text-xs text-slate-500">
        Крива капіталу недоступна (запустіть або оберіть завершений бектест).
      </div>
    );
  }

  return (
    <div className="w-full rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
      <div ref={containerRef} className="w-full h-[300px]" />
    </div>
  );
}

export function BacktestLab() {
  const queryClient = useQueryClient();
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState<"1h" | "1d">("1h");
  const [mode, setMode] = useState<"walk_forward_retrain" | "static_model">(
    "walk_forward_retrain"
  );
  const [periodFrom, setPeriodFrom] = useState("2024-11-18");
  const [periodTo, setPeriodTo] = useState("2026-11-18");
  const [entryThreshold, setEntryThreshold] = useState("0.55");
  const [positionSize, setPositionSize] = useState("1.0");
  const [allowShort, setAllowShort] = useState(false);
  const [selectedBtId, setSelectedBtId] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [launchingBtId, setLaunchingBtId] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const instrumentsQuery = useQuery({
    queryKey: ["instruments"],
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/instruments");
      return res.data ?? [];
    },
  });
  const instrumentOptions =
    instrumentsQuery.data && instrumentsQuery.data.length > 0
      ? instrumentsQuery.data.map((i) => i.symbol)
      : ["BTCUSDT", "ETHUSDT", "SPY"];

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/models");
      return res.data ?? [];
    },
  });

  const backtestsQuery = useQuery({
    queryKey: ["backtests"],
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/backtests");
      return res.data ?? [];
    },
  });

  const activeBtId = selectedBtId ?? backtestsQuery.data?.[0]?.id ?? null;

  const jobQuery = useQuery({
    queryKey: ["job", activeJobId],
    enabled: activeJobId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/jobs/{id}", {
        params: { path: { id: activeJobId! } },
      });
      return res.data;
    },
    refetchInterval: (query) => {
      const st = query.state.data?.status;
      if (st === "done" || st === "failed" || st === "cancelled") {
        return false;
      }
      return 2000;
    },
  });

  useEffect(() => {
    const st = jobQuery.data?.status;
    if (st === "done" || st === "failed") {
      const targetId = launchingBtId ?? activeBtId;
      void queryClient.invalidateQueries({ queryKey: ["backtests"] });
      if (targetId) {
        void queryClient.invalidateQueries({ queryKey: ["backtest-detail", targetId] });
        void queryClient.invalidateQueries({ queryKey: ["backtest-equity", targetId] });
        void queryClient.invalidateQueries({ queryKey: ["backtest-folds", targetId] });
        void queryClient.invalidateQueries({ queryKey: ["backtest-trades", targetId] });
      }
      setLaunchingBtId(null);
      setActiveJobId(null);
    }
  }, [jobQuery.data?.status, launchingBtId, activeBtId, queryClient]);

  const detailQuery = useQuery({
    queryKey: ["backtest-detail", activeBtId],
    enabled: activeBtId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/backtests/{id}", {
        params: { path: { id: activeBtId! } },
      });
      return res.data;
    },
  });

  const equityQuery = useQuery({
    queryKey: ["backtest-equity", activeBtId],
    enabled: activeBtId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/backtests/{id}/equity", {
        params: { path: { id: activeBtId! } },
      });
      return res.data;
    },
  });

  const foldsQuery = useQuery({
    queryKey: ["backtest-folds", activeBtId],
    enabled: activeBtId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/backtests/{id}/folds", {
        params: { path: { id: activeBtId! } },
      });
      return res.data ?? [];
    },
  });

  const tradesQuery = useQuery({
    queryKey: ["backtest-trades", activeBtId],
    enabled: activeBtId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/backtests/{id}/trades", {
        params: { path: { id: activeBtId! }, query: { limit: 50, offset: 0 } },
      });
      return res.data;
    },
  });

  const createMutation = useMutation({
    mutationFn: async () => {
      const res = await apiClient.POST("/api/v1/backtests", {
        body: {
          name: `${symbol} ${timeframe} ${mode === "walk_forward_retrain" ? "Walk-Forward" : "Static OOS"}`,
          instrument_ids: [symbol],
          timeframe,
          period: { from: periodFrom, to: periodTo },
          mode,
          strategy: {
            entry_threshold: Number(entryThreshold) || 0.55,
            min_margin: 0.1,
            exit_mode: "label_barriers",
            position_size: Math.min(1, Math.max(0.1, Number(positionSize) || 1.0)),
            max_holding_bars: timeframe === "1h" ? 24 : 5,
            pt_mult: 1.0,
            sl_mult: 1.0,
            allow_short: allowShort,
          },
          costs: { fee_bps: 10.0, slippage_bps: 5.0 },
        },
      });
      return res.data;
    },
    onSuccess: (data) => {
      setFormError(null);
      if (data) {
        setSelectedBtId(data.backtest_id);
        setLaunchingBtId(data.backtest_id);
        setActiveJobId(data.job_id);
        void queryClient.invalidateQueries({ queryKey: ["backtests"] });
      }
    },
    onError: (err) => {
      if (err instanceof ApiError || err instanceof Error) {
        setFormError(err.message);
      } else {
        setFormError("Не вдалося запустити бектест");
      }
    },
  });

  const metricsObj = (detailQuery.data?.metrics ?? {}) as Record<string, BacktestMetricDict>;
  const strat = metricsObj.strategy ?? {};
  const bh = metricsObj.buy_and_hold ?? {};
  const ema = metricsObj.ema_crossover ?? {};
  const rand = metricsObj.random_benchmark ?? {};

  return (
    <div className="flex-1 p-6 grid grid-cols-12 gap-6">
      {/* Left: Backtest Launch Form & Run History */}
      <section className="col-span-12 lg:col-span-4 xl:col-span-3 space-y-5">
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 border-b border-slate-800 pb-2">
            <Play className="w-4 h-4 text-indigo-400" />
            Новий бектест (label_barriers)
          </h2>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <label className="space-y-1">
              <span className="text-slate-400">Актив</span>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
              >
                {instrumentOptions.map((sym) => (
                  <option key={sym} value={sym}>
                    {sym}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1">
              <span className="text-slate-400">Режим</span>
              <select
                value={mode}
                onChange={(e) =>
                  setMode(e.target.value as "walk_forward_retrain" | "static_model")
                }
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
              >
                <option value="walk_forward_retrain">Walk-Forward</option>
                <option value="static_model">Static OOS</option>
              </select>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <label className="space-y-1">
              <span className="text-slate-400">Від</span>
              <input
                type="date"
                value={periodFrom}
                onChange={(e) => setPeriodFrom(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
              />
            </label>
            <label className="space-y-1">
              <span className="text-slate-400">До</span>
              <input
                type="date"
                value={periodTo}
                onChange={(e) => setPeriodTo(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
              />
            </label>
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs">
            <label className="space-y-1">
              <span className="text-slate-400">Поріг входу</span>
              <input
                type="number"
                step="0.05"
                min="0.35"
                max="0.90"
                value={entryThreshold}
                onChange={(e) => setEntryThreshold(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
              />
            </label>
            <label className="space-y-1">
              <span className="text-slate-400">Таймфрейм</span>
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value as "1h" | "1d")}
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
              >
                <option value="1h">1h</option>
                <option value="1d">1d</option>
              </select>
            </label>
            <label className="space-y-1">
              <span className="text-slate-400">Розмір поз.</span>
              <select
                value={positionSize}
                onChange={(e) => setPositionSize(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
              >
                <option value="1.0">100%</option>
                <option value="0.5">50%</option>
                <option value="0.25">25%</option>
              </select>
            </label>
          </div>
          <label className="flex items-center gap-2 text-xs cursor-pointer text-slate-300">
            <input
              type="checkbox"
              checked={allowShort}
              onChange={(e) => setAllowShort(e.target.checked)}
              className="rounded border-slate-700"
            />
            Дозволити SHORT-позиції (Perpetual режим)
          </label>

          <div className="text-[11px] text-slate-500">
            Виконання: <span className="font-mono">open[t+1]</span> · Витрати: 10 bps + 5 bps
          </div>

          {formError && (
            <div className="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-[11px] text-rose-300 flex gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{formError}</span>
            </div>
          )}

          {activeJobId && jobQuery.data && (
            <div className="p-2.5 rounded bg-indigo-500/10 border border-indigo-500/30 text-xs space-y-1">
              <div className="flex justify-between text-indigo-300">
                <span>Виконання ({jobQuery.data.status})</span>
                <span className="font-mono">{(jobQuery.data.progress * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 transition-all"
                  style={{ width: `${jobQuery.data.progress * 100}%` }}
                />
              </div>
            </div>
          )}

          <button
            onClick={() => createMutation.mutate()}
            disabled={createMutation.isPending || activeJobId !== null}
            className="w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white transition-colors"
          >
            {createMutation.isPending ? "Запуск..." : "Запустити бектест"}
          </button>
        </div>

        {/* Backtest History */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3">
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Історія запусків ({backtestsQuery.data?.length ?? 0})
          </h3>
          <div className="space-y-2 max-h-[280px] overflow-y-auto">
            {(backtestsQuery.data ?? []).map((bt) => (
              <button
                key={bt.id}
                onClick={() => setSelectedBtId(bt.id)}
                className={`w-full text-left p-2.5 rounded-lg border text-xs transition-colors ${
                  activeBtId === bt.id
                    ? "bg-indigo-600/20 border-indigo-500/50 text-white"
                    : "bg-slate-950/50 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-semibold truncate">{bt.name}</span>
                  <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-slate-800">
                    {bt.status}
                  </span>
                </div>
                <div className="flex justify-between text-[11px] text-slate-500 mt-1 font-mono">
                  <span>Sharpe: {bt.sharpe != null ? bt.sharpe.toFixed(2) : "—"}</span>
                  <span>
                    Ret: {bt.total_return != null ? `${(bt.total_return * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Right: Equity Curve, Metrics vs Baselines, Folds, Trades, Models */}
      <section className="col-span-12 lg:col-span-8 xl:col-span-9 space-y-5">
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-emerald-400" />
                {detailQuery.data?.name ?? "Крива капіталу та метрики"}
              </h2>
              <p className="text-xs text-slate-400">
                Зелена: LGBM Strategy · Синя: Buy & Hold · Помаранчева: EMA 20/50
              </p>
            </div>
            {detailQuery.data && (
              <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                {detailQuery.data.mode} · {detailQuery.data.status}
              </span>
            )}
          </div>

          <EquityCurveChart items={equityQuery.data?.items ?? []} />

          {/* Strategy KPI Strip */}
          {typeof strat.sharpe === "number" && (
            <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-6 gap-2.5 text-xs">
              {[
                {
                  label: "Total Return",
                  val: `${((strat.total_return as number) * 100).toFixed(2)}%`,
                },
                { label: "Sharpe", val: (strat.sharpe as number).toFixed(2) },
                { label: "Sortino", val: ((strat.sortino as number) ?? 0).toFixed(2) },
                {
                  label: "Max Drawdown",
                  val: `${(((strat.max_drawdown as number) ?? 0) * 100).toFixed(2)}%`,
                },
                { label: "Profit Factor", val: ((strat.profit_factor as number) ?? 0).toFixed(2) },
                {
                  label: "Win Rate / Trades",
                  val: `${(((strat.win_rate as number) ?? 0) * 100).toFixed(1)}% (${strat.trades_count ?? 0})`,
                },
              ].map((card) => (
                <div
                  key={card.label}
                  className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800"
                >
                  <div className="text-slate-500 text-[11px]">{card.label}</div>
                  <div className="font-mono font-bold text-slate-100 mt-0.5">{card.val}</div>
                </div>
              ))}
            </div>
          )}

          {/* Strategy vs Baselines Comparison */}
          {typeof strat.sharpe === "number" && (
            <div className="space-y-3">
              {(metricsObj.acceptance_a3 || (Array.isArray(detailQuery.data?.metrics?.warnings) && detailQuery.data?.metrics?.warnings.length > 0)) && (
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  {metricsObj.acceptance_a3 && (
                    <>
                      <span
                        className={`px-2 py-0.5 rounded border font-mono ${
                          metricsObj.acceptance_a3.sharpe_gt_zero
                            ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                            : "bg-amber-500/10 text-amber-300 border-amber-500/30"
                        }`}
                      >
                        Sharpe &gt; 0: {metricsObj.acceptance_a3.sharpe_gt_zero ? "PASS" : "FAIL"}
                      </span>
                      <span className="px-2 py-0.5 rounded border bg-slate-800/80 text-slate-300 border-slate-700 font-mono">
                        Фолди &gt; B&amp;H: {String(metricsObj.acceptance_a3.folds_beating_bh_sharpe ?? "—")}
                      </span>
                    </>
                  )}
                  {Array.isArray(detailQuery.data?.metrics?.warnings) &&
                    (detailQuery.data.metrics.warnings as string[]).map((w) => (
                      <span
                        key={w}
                        className="px-2 py-0.5 rounded border bg-rose-500/10 text-rose-300 border-rose-500/30 font-mono"
                      >
                        {w}
                      </span>
                    ))}
                </div>
              )}
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2 pr-3">Стратегія</th>
                    <th className="py-2 px-3 text-right">Return</th>
                    <th className="py-2 px-3 text-right">Sharpe</th>
                    <th className="py-2 px-3 text-right">MaxDD</th>
                    <th className="py-2 pl-3 text-right">Trades</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {[
                    { name: "LGBM (label_barriers)", row: strat },
                    { name: "Buy & Hold", row: bh },
                    { name: "EMA(20/50) Crossover", row: ema },
                  ].map(({ name, row }) => (
                    <tr key={name} className="text-slate-200">
                      <td className="py-2 pr-3 font-sans font-medium">{name}</td>
                      <td className="py-2 px-3 text-right">
                        {typeof row.total_return === "number"
                          ? `${(row.total_return * 100).toFixed(2)}%`
                          : "—"}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {typeof row.sharpe === "number" ? row.sharpe.toFixed(2) : "—"}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {typeof row.max_drawdown === "number"
                          ? `${(row.max_drawdown * 100).toFixed(2)}%`
                          : "—"}
                      </td>
                      <td className="py-2 pl-3 text-right">{row.trades_count ?? "—"}</td>
                    </tr>
                  ))}
                  {typeof rand.sharpe_mean === "number" && (
                    <tr className="text-slate-400">
                      <td className="py-2 pr-3 font-sans">
                        Random ({String(rand.n_seeds ?? 20)} seeds)
                      </td>
                      <td className="py-2 px-3 text-right">
                        {(((rand.total_return_mean as number) ?? 0) * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-right">
                        {(rand.sharpe_mean as number).toFixed(2)} ±{" "}
                        {((rand.sharpe_std as number) ?? 0).toFixed(2)}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {(((rand.max_drawdown_mean as number) ?? 0) * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 pl-3 text-right">—</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            </div>
          )}
        </div>

        {/* Registered Models Table */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-3">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-indigo-400" />
            Реєстр моделей (`model_versions`)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-3">Модель</th>
                  <th className="py-2 px-2">Актив</th>
                  <th className="py-2 px-2">Статус</th>
                  <th className="py-2 px-2 text-right">Macro-F1</th>
                  <th className="py-2 px-2 text-right">ECE-10</th>
                  <th className="py-2 pl-2 text-right">Train End</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {(modelsQuery.data ?? []).map((mv) => {
                  const mObj = mv.metrics as Record<string, Record<string, Record<string, Record<string, number>>>>;
                  const oosLgbm = mObj?.oos_classification?.models?.lgbm_calibrated;
                  return (
                    <tr key={mv.id} className="text-slate-300">
                      <td className="py-2 pr-3 font-mono">
                        {mv.name} v{mv.semver}
                      </td>
                      <td className="py-2 px-2 font-mono">
                        {mv.symbol ?? "—"} {mv.timeframe ?? ""}
                      </td>
                      <td className="py-2 px-2">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold ${
                            mv.status === "production"
                              ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                              : "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                          }`}
                        >
                          {mv.status === "production" && <CheckCircle2 className="w-3 h-3" />}
                          {mv.status}
                        </span>
                      </td>
                      <td className="py-2 px-2 text-right font-mono">
                        {typeof oosLgbm?.macro_f1 === "number" ? oosLgbm.macro_f1.toFixed(4) : "—"}
                      </td>
                      <td className="py-2 px-2 text-right font-mono">
                        {typeof oosLgbm?.ece_10 === "number" ? oosLgbm.ece_10.toFixed(4) : "—"}
                      </td>
                      <td className="py-2 pl-2 text-right font-mono text-slate-400">
                        {formatDateTime(mv.train_data_end)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Walk-Forward Folds & Recent Trades */}
        {(foldsQuery.data?.length ?? 0) > 0 && (
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Activity className="w-4 h-4 text-sky-400" />
              Фолди Purged Walk-Forward ({foldsQuery.data?.length ?? 0})
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-sans">
                    <th className="py-1.5 pr-2">Fold</th>
                    <th className="py-1.5 px-2">Test період</th>
                    <th className="py-1.5 px-2 text-right">Return</th>
                    <th className="py-1.5 px-2 text-right">Sharpe</th>
                    <th className="py-1.5 px-2 text-right">WinRate</th>
                    <th className="py-1.5 pl-2 text-right">Trades</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {(foldsQuery.data ?? []).map((f) => {
                    const fm = f.metrics as Record<string, number>;
                    return (
                      <tr key={f.fold_no} className="text-slate-300">
                        <td className="py-1.5 pr-2">#{f.fold_no}</td>
                        <td className="py-1.5 px-2 text-slate-400">
                          {f.test_start.slice(0, 10)} → {f.test_end.slice(0, 10)}
                        </td>
                        <td className="py-1.5 px-2 text-right">
                          {typeof fm.total_return === "number"
                            ? `${(fm.total_return * 100).toFixed(2)}%`
                            : "—"}
                        </td>
                        <td className="py-1.5 px-2 text-right">
                          {typeof fm.sharpe === "number" ? fm.sharpe.toFixed(2) : "—"}
                        </td>
                        <td className="py-1.5 px-2 text-right">
                          {typeof fm.win_rate === "number"
                            ? `${(fm.win_rate * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className="py-1.5 pl-2 text-right">{fm.trades_count ?? 0}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {(tradesQuery.data?.items?.length ?? 0) > 0 && (
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200">
                Останні угоди ({tradesQuery.data?.total ?? 0} всього)
              </h3>
              <button
                onClick={() =>
                  downloadTradesCsv(
                    detailQuery.data?.name ?? "backtest",
                    tradesQuery.data?.items ?? []
                  )
                }
                className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
              >
                <Download className="w-3.5 h-3.5" />
                CSV
              </button>
            </div>
            <div className="overflow-x-auto max-h-[260px]">
              <table className="w-full text-xs text-left border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-sans">
                    <th className="py-1.5 pr-2">#</th>
                    <th className="py-1.5 px-2">Side</th>
                    <th className="py-1.5 px-2">Entry</th>
                    <th className="py-1.5 px-2">Exit</th>
                    <th className="py-1.5 px-2">Reason</th>
                    <th className="py-1.5 pl-2 text-right">Net PnL</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {(tradesQuery.data?.items ?? []).map((t) => (
                    <tr key={t.seq} className="text-slate-300">
                      <td className="py-1.5 pr-2">{t.seq}</td>
                      <td className="py-1.5 px-2">{t.side}</td>
                      <td className="py-1.5 px-2">
                        {t.entry_at.slice(0, 16).replace("T", " ")} @ {t.entry_price.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-2">
                        {t.exit_at.slice(0, 16).replace("T", " ")} @ {t.exit_price.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-2">{t.exit_reason}</td>
                      <td
                        className={`py-1.5 pl-2 text-right ${
                          t.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {t.pnl >= 0 ? "+" : ""}
                        {t.pnl.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
