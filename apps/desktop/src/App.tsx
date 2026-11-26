import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError } from "@finsight/api-client";
import { apiClient } from "@/lib/api";
import { formatCurrency, formatDateTime, formatNumber } from "@finsight/shared";
import {
  MarketChart,
  type IndicatorToggles,
} from "@/components/MarketChart";
import { ForecastPanel } from "@/components/ForecastPanel";
import { BacktestLab } from "@/components/BacktestLab";
import { PortfolioRiskLab } from "@/components/PortfolioRiskLab";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Briefcase,
  Clock,
  FlaskConical,
  RefreshCw,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

type SupportedTimeframe = "1h" | "1d";
type ActiveTab = "market" | "backtest" | "portfolio";

const INDICATOR_CONTROLS: ReadonlyArray<{
  key: keyof IndicatorToggles;
  label: string;
  dotColor: string;
}> = [
  { key: "ema20", label: "EMA 20", dotColor: "bg-amber-400" },
  { key: "ema50", label: "EMA 50", dotColor: "bg-cyan-400" },
  { key: "ema200", label: "EMA 200", dotColor: "bg-purple-400" },
  { key: "bb", label: "Bollinger", dotColor: "bg-indigo-400" },
  { key: "rsi14", label: "RSI 14", dotColor: "bg-sky-400" },
  { key: "macd", label: "MACD", dotColor: "bg-teal-400" },
];

export default function App() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<ActiveTab>("market");
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState<SupportedTimeframe>("1h");
  const [showSignals, setShowSignals] = useState(true);
  const [toggles, setToggles] = useState<IndicatorToggles>({
    ema20: true,
    ema50: true,
    ema200: false,
    bb: false,
    rsi14: true,
    macd: false,
  });
  const [refreshFeedback, setRefreshFeedback] = useState<string | null>(null);

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const res = await apiClient.GET("/health");
      return res.data;
    },
    refetchInterval: 30_000,
  });

  const instrumentsQuery = useQuery({
    queryKey: ["instruments"],
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/instruments");
      return res.data ?? [];
    },
  });

  const quoteQuery = useQuery({
    queryKey: ["quote", selectedSymbol],
    enabled: activeTab === "market",
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/instruments/{id}/quote", {
        params: { path: { id: selectedSymbol } },
      });
      return res.data;
    },
    refetchInterval: 30_000,
  });

  const candlesQuery = useQuery({
    queryKey: ["candles", selectedSymbol, timeframe],
    enabled: activeTab === "market",
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/instruments/{id}/candles", {
        params: {
          path: { id: selectedSymbol },
          query: { timeframe, limit: 500 },
        },
      });
      return res.data;
    },
  });

  const indicatorsQuery = useQuery({
    queryKey: ["indicators", selectedSymbol, timeframe],
    enabled: activeTab === "market",
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/instruments/{id}/indicators", {
        params: {
          path: { id: selectedSymbol },
          query: { timeframe, limit: 500 },
        },
      });
      return res.data;
    },
  });

  const predictionHistoryQuery = useQuery({
    queryKey: ["prediction-history", selectedSymbol, timeframe],
    enabled: activeTab === "market",
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/predictions/history", {
        params: {
          query: { instrument_id: selectedSymbol, timeframe, limit: 300 },
        },
      });
      return res.data ?? [];
    },
  });

  const regimeQuery = useQuery({
    queryKey: ["regime", selectedSymbol, timeframe],
    enabled: activeTab === "market",
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/instruments/{id}/regime", {
        params: { path: { id: selectedSymbol }, query: { timeframe, days: 30 } },
      });
      return res.data;
    },
  });

  const portfoliosQuery = useQuery({
    queryKey: ["portfolios"],
    enabled: activeTab === "market",
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/portfolios");
      return res.data ?? [];
    },
  });

  const primaryPortfolioId = portfoliosQuery.data?.[0]?.id ?? null;

  const sidebarRiskQuery = useQuery({
    queryKey: ["portfolio-risk", primaryPortfolioId, "90d"],
    enabled: activeTab === "market" && primaryPortfolioId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/portfolios/{id}/risk", {
        params: { path: { id: primaryPortfolioId! }, query: { window: "90d" } },
      });
      return res.data;
    },
  });

  const refreshMutation = useMutation({
    mutationFn: async (vars: { symbol: string; tf: SupportedTimeframe }) => {
      const res = await apiClient.POST("/api/v1/instruments/{id}/refresh", {
        params: {
          path: { id: vars.symbol },
          query: { timeframe: vars.tf },
        },
      });
      return res.data;
    },
    onSuccess: async (data, vars) => {
      setRefreshFeedback(
        `Оновлено ${data?.rows_written ?? 0} барів (${vars.symbol} ${vars.tf})`
      );
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["quote", vars.symbol] }),
        queryClient.invalidateQueries({ queryKey: ["candles", vars.symbol, vars.tf] }),
        queryClient.invalidateQueries({ queryKey: ["indicators", vars.symbol, vars.tf] }),
      ]);
    },
    onError: (err) => {
      if (err instanceof ApiError || err instanceof Error) {
        setRefreshFeedback(err.message);
      } else {
        setRefreshFeedback("Помилка оновлення даних");
      }
    },
  });

  const toggleIndicator = (key: keyof IndicatorToggles) => {
    setToggles((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const selectedInstrument = instrumentsQuery.data?.find(
    (inst) => inst.symbol === selectedSymbol
  );
  const candles = candlesQuery.data?.items ?? [];
  const indicators = indicatorsQuery.data?.items ?? [];
  const signalHistory = predictionHistoryQuery.data ?? [];
  const latestIndicator = indicators[indicators.length - 1];
  const quote = quoteQuery.data;
  const isPositiveChange = (quote?.change_24h ?? 0) >= 0;

  const chartError =
    candlesQuery.error instanceof Error
      ? candlesQuery.error.message
      : indicatorsQuery.error instanceof Error
        ? indicatorsQuery.error.message
        : null;

  return (
    <div className="flex flex-col min-h-screen w-full bg-slate-950 text-slate-100 font-sans">
      {/* Top Navbar */}
      <header className="flex flex-wrap items-center justify-between gap-4 px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              FinSight AI
              <span className="text-xs font-medium px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                v1.1 · C3 Ризик і портфель
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Кросплатформна інтелектуальна система аналізу фінансових активів
            </p>
          </div>
        </div>

        {/* Navigation Tabs + API Status */}
        <div className="flex items-center gap-4">
          <nav className="flex bg-slate-800/80 p-1 rounded-lg border border-slate-700 space-x-1 text-xs">
            <button
              onClick={() => setActiveTab("market")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors ${
                activeTab === "market"
                  ? "bg-indigo-600 text-white font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              Ринок і прогноз
            </button>
            <button
              onClick={() => setActiveTab("backtest")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors ${
                activeTab === "backtest"
                  ? "bg-indigo-600 text-white font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <FlaskConical className="w-3.5 h-3.5" />
              Backtest Lab
            </button>
            <button
              onClick={() => setActiveTab("portfolio")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors ${
                activeTab === "portfolio"
                  ? "bg-indigo-600 text-white font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Briefcase className="w-3.5 h-3.5" />
              Портфель і ризик
            </button>
          </nav>

          <div className="flex items-center space-x-2 text-xs bg-slate-800/80 px-3 py-1.5 rounded-full border border-slate-700">
            <div
              className={`w-2 h-2 rounded-full ${
                healthQuery.data?.status === "ok"
                  ? "bg-emerald-400 animate-pulse"
                  : "bg-amber-400"
              }`}
            />
            <span className="text-slate-300">
              API:{" "}
              {healthQuery.isLoading
                ? "перевірка..."
                : healthQuery.data?.status === "ok"
                  ? "онлайн"
                  : "офлайн"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === "backtest" ? (
        <BacktestLab />
      ) : activeTab === "portfolio" ? (
        <PortfolioRiskLab />
      ) : (
        <main className="flex-1 p-6 grid grid-cols-12 gap-6">
          {/* Left Sidebar: Instruments */}
          <section className="col-span-12 lg:col-span-3 xl:col-span-2 bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-indigo-400" />
                Активи
              </h2>
              <span className="text-xs text-slate-500">
                {instrumentsQuery.data?.length ?? 0} шт.
              </span>
            </div>

            <div className="space-y-2">
              {instrumentsQuery.isLoading ? (
                <div className="text-xs text-slate-500">Завантаження активів...</div>
              ) : instrumentsQuery.isError ? (
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>Немає зв&apos;язку з API. Перевірте бекенд на порту 8000.</span>
                </div>
              ) : (
                instrumentsQuery.data?.map((inst) => (
                  <button
                    key={inst.id}
                    onClick={() => {
                      setSelectedSymbol(inst.symbol);
                      setRefreshFeedback(null);
                    }}
                    className={`w-full flex items-center justify-between p-3 rounded-lg text-left transition-all ${
                      selectedSymbol === inst.symbol
                        ? "bg-indigo-600/20 border border-indigo-500/50 text-white"
                        : "bg-slate-800/40 hover:bg-slate-800/70 border border-transparent text-slate-400"
                    }`}
                  >
                    <div>
                      <div className="font-semibold text-sm text-slate-200">{inst.symbol}</div>
                      <div className="text-xs text-slate-500 truncate max-w-[120px]">
                        {inst.name}
                      </div>
                    </div>
                    <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      {inst.asset_class}
                    </span>
                  </button>
                ))
              )}
            </div>

            {selectedInstrument && (
              <div className="mt-auto pt-3 border-t border-slate-800/80 text-xs text-slate-400 space-y-1">
                <div className="flex justify-between">
                  <span>Біржа:</span>
                  <span className="uppercase font-mono text-slate-300">
                    {selectedInstrument.exchange}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Валюта:</span>
                  <span className="font-mono text-slate-300">
                    {selectedInstrument.quote_currency}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Бари на графіку:</span>
                  <span className="font-mono text-slate-300">{candles.length}</span>
                </div>
              </div>
            )}
          </section>

          {/* Center: Live Chart Workspace */}
          <section className="col-span-12 lg:col-span-9 xl:col-span-7 bg-slate-900/40 border border-slate-800 rounded-xl p-5 flex flex-col space-y-4">
            {/* Quote Header + Controls */}
            <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div>
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-2xl font-bold text-white">{selectedSymbol}</span>
                  {quote ? (
                    <>
                      <span className="text-xl font-mono font-semibold text-slate-100">
                        {formatCurrency(quote.price, "USD", "en-US")}
                      </span>
                      <span
                        className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded border ${
                          isPositiveChange
                            ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                            : "bg-rose-500/15 text-rose-400 border-rose-500/30"
                        }`}
                      >
                        {isPositiveChange ? (
                          <TrendingUp className="w-3.5 h-3.5" />
                        ) : (
                          <TrendingDown className="w-3.5 h-3.5" />
                        )}
                        {isPositiveChange ? "+" : ""}
                        {quote.change_24h.toFixed(2)}% (24г)
                      </span>
                      {quote.stale && (
                        <span
                          title="Дані старші за очікуваний бар з урахуванням розкладу біржі"
                          className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30"
                        >
                          <Clock className="w-3 h-3" />
                          Застаріло
                        </span>
                      )}
                    </>
                  ) : quoteQuery.isLoading ? (
                    <span className="text-sm text-slate-500">Завантаження котирування...</span>
                  ) : null}
                </div>

                <div className="text-xs text-slate-400 mt-1 flex flex-wrap items-center gap-3">
                  <span>
                    Останній закритий бар:{" "}
                    <span className="font-mono text-slate-300">
                      {quote?.ts ? formatDateTime(quote.ts) : "—"}
                    </span>
                  </span>
                  {quote && (
                    <span>
                      Обсяг (24г):{" "}
                      <span className="font-mono text-slate-300">
                        {formatNumber(quote.volume_24h, "en-US", 1)}
                      </span>
                    </span>
                  )}
                </div>
              </div>

              {/* Timeframe Selector + Refresh Button */}
              <div className="flex items-center gap-2">
                <div className="flex bg-slate-800/80 p-1 rounded-lg border border-slate-700 space-x-1">
                  {(["1h", "1d"] as const).map((tf) => (
                    <button
                      key={tf}
                      onClick={() => {
                        setTimeframe(tf);
                        setRefreshFeedback(null);
                      }}
                      className={`text-xs px-3 py-1 rounded transition-colors ${
                        timeframe === tf
                          ? "bg-indigo-600 text-white font-medium"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>

                <button
                  onClick={() => refreshMutation.mutate({ symbol: selectedSymbol, tf: timeframe })}
                  disabled={refreshMutation.isPending}
                  title="Оновити свічки з біржі (ліміт 1 раз / 60с)"
                  className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 border border-slate-700 transition-colors"
                >
                  <RefreshCw
                    className={`w-3.5 h-3.5 ${refreshMutation.isPending ? "animate-spin" : ""}`}
                  />
                  Оновити
                </button>
              </div>
            </div>

            {/* Indicator & Signal Toggles Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                {INDICATOR_CONTROLS.map(({ key, label, dotColor }) => {
                  const active = toggles[key];
                  return (
                    <button
                      key={key}
                      onClick={() => toggleIndicator(key)}
                      className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md border transition-colors ${
                        active
                          ? "bg-slate-800 text-slate-100 border-indigo-500/60"
                          : "bg-slate-900/50 text-slate-500 border-slate-800 hover:text-slate-300"
                      }`}
                    >
                      <span
                        className={`w-2 h-2 rounded-full ${active ? dotColor : "bg-slate-600"}`}
                      />
                      {label}
                    </button>
                  );
                })}

                <button
                  onClick={() => setShowSignals((prev) => !prev)}
                  className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md border transition-colors ${
                    showSignals
                      ? "bg-emerald-500/15 text-emerald-200 border-emerald-500/50"
                      : "bg-slate-900/50 text-slate-500 border-slate-800 hover:text-slate-300"
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      showSignals ? "bg-emerald-400" : "bg-slate-600"
                    }`}
                  />
                  Сигнали ({signalHistory.filter((s) => s.signal !== "HOLD").length})
                </button>
              </div>

              {refreshFeedback && (
                <span className="text-[11px] text-indigo-300 bg-indigo-500/10 border border-indigo-500/30 px-2.5 py-1 rounded">
                  {refreshFeedback}
                </span>
              )}
            </div>

            {/* Lightweight Charts v5 Live Multi-Pane Chart */}
            <MarketChart
              candles={candles}
              indicators={indicators}
              signals={signalHistory}
              showSignals={showSignals}
              toggles={toggles}
              isLoading={candlesQuery.isLoading || indicatorsQuery.isLoading}
              errorMessage={chartError}
            />

            {/* Live Indicator Readout Strip */}
            {latestIndicator && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500">EMA 20 / 50</div>
                  <div className="font-mono text-slate-200 mt-0.5">
                    {latestIndicator.ema20 != null ? latestIndicator.ema20.toFixed(2) : "—"} /{" "}
                    {latestIndicator.ema50 != null ? latestIndicator.ema50.toFixed(2) : "—"}
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500">EMA 200</div>
                  <div className="font-mono text-purple-300 mt-0.5">
                    {latestIndicator.ema200 != null ? latestIndicator.ema200.toFixed(2) : "—"}
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500">RSI 14</div>
                  <div className="font-mono text-sky-300 mt-0.5">
                    {latestIndicator.rsi14 != null ? latestIndicator.rsi14.toFixed(2) : "—"}
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500">ATR 14</div>
                  <div className="font-mono text-slate-200 mt-0.5">
                    {latestIndicator.atr14 != null ? latestIndicator.atr14.toFixed(2) : "—"}
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* Right Sidebar: Live C2 Forecast + HMM Regime + C3 Portfolio Risk */}
          <section className="col-span-12 xl:col-span-3 space-y-6">
            <ForecastPanel
              symbol={selectedSymbol}
              timeframe={timeframe}
              currentPrice={quote?.price}
            />

            {/* Causal 3-state HMM Regime Card */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-sky-400" />
                  Режим ринку (HMM)
                </h2>
                <span
                  className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded border ${
                    regimeQuery.data?.regime === "bull"
                      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                      : regimeQuery.data?.regime === "bear"
                        ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
                        : "bg-slate-800 text-slate-300 border-slate-700"
                  }`}
                >
                  {regimeQuery.data?.regime ?? "—"}
                </span>
              </div>
              {regimeQuery.data ? (
                <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                  {(["bull", "range", "bear"] as const).map((st) => (
                    <div key={st} className="p-2 rounded bg-slate-950/60 border border-slate-800">
                      <div className="text-[10px] uppercase text-slate-500">{st}</div>
                      <div className="text-slate-200 font-semibold mt-0.5">
                        {((regimeQuery.data?.probabilities[st] ?? 0) * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-slate-500">Завантаження режиму...</div>
              )}
            </div>

            <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-indigo-400" />
                  Ризик портфеля
                </h2>
                <button
                  onClick={() => setActiveTab("portfolio")}
                  className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/15 hover:bg-indigo-500/25 text-indigo-300 border border-indigo-500/30"
                >
                  Деталі →
                </button>
              </div>
              {sidebarRiskQuery.data ? (
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                    <div className="text-slate-500">VaR (95%)</div>
                    <div className="text-base font-bold text-amber-300 mt-1 font-mono">
                      {(sidebarRiskQuery.data.var_95 * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                    <div className="text-slate-500">Sharpe Ratio</div>
                    <div className="text-base font-bold text-emerald-400 mt-1 font-mono">
                      {sidebarRiskQuery.data.sharpe.toFixed(2)}
                    </div>
                  </div>
                  <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                    <div className="text-slate-500">Max Drawdown</div>
                    <div className="text-base font-bold text-amber-400 mt-1 font-mono">
                      {(sidebarRiskQuery.data.max_drawdown * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                    <div className="text-slate-500">Рівень ризику</div>
                    <div className="text-base font-bold text-indigo-400 mt-1 capitalize">
                      {sidebarRiskQuery.data.risk_level}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-500">
                  {portfoliosQuery.data?.length === 0
                    ? "Створіть портфель у вкладці «Портфель і ризик»."
                    : "Очікування даних ризику портфеля..."}
                </div>
              )}
            </div>
          </section>
        </main>
      )}

      {/* Footer */}
      <footer className="px-6 py-3 border-t border-slate-800/80 bg-slate-900/30 text-xs text-slate-500 flex justify-between items-center">
        <span>Курсова робота → Дипломний проєкт · Спеціальність 122 «Комп&apos;ютерні науки»</span>
        <span>Lightweight Charts v5 · LightGBM + Purged Walk-Forward</span>
      </footer>
    </div>
  );
}
