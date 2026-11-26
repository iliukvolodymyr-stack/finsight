import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError } from "@finsight/api-client";
import { apiClient } from "@/lib/api";
import { formatCurrency, formatDateTime } from "@finsight/shared";
import {
  Activity,
  AlertCircle,
  Briefcase,
  Grid3X3,
  Plus,
  RotateCcw,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

function corrCellStyle(val: number): { backgroundColor: string; color: string } {
  const clamped = Math.max(-1, Math.min(1, val));
  const abs = Math.abs(clamped);
  if (clamped >= 0) {
    const alpha = 0.14 + abs * 0.55;
    return {
      backgroundColor: `rgba(16, 185, 129, ${alpha.toFixed(2)})`,
      color: abs > 0.55 ? "#ecfdf5" : "#a7f3d0",
    };
  }
  const alpha = 0.14 + abs * 0.55;
  return {
    backgroundColor: `rgba(244, 63, 94, ${alpha.toFixed(2)})`,
    color: abs > 0.55 ? "#fff1f2" : "#fecdd3",
  };
}

function RiskLevelBadge({ level }: { level: string }) {
  const styles =
    level === "low"
      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
      : level === "high"
        ? "bg-rose-500/15 text-rose-300 border-rose-500/40"
        : "bg-amber-500/15 text-amber-300 border-amber-500/40";
  const label = level === "low" ? "Низький" : level === "high" ? "Високий" : "Помірний";
  return (
    <span className={`text-xs font-bold uppercase px-2.5 py-0.5 rounded border ${styles}`}>
      {label} ({level})
    </span>
  );
}

export function PortfolioRiskLab() {
  const queryClient = useQueryClient();
  const [selectedPortId, setSelectedPortId] = useState<string | null>(null);
  const [corrWindow, setCorrWindow] = useState<"30d" | "90d" | "180d">("90d");
  const [newPortName, setNewPortName] = useState("");
  const [showNewPort, setShowNewPort] = useState(false);

  // Transaction form state
  const [txSymbol, setTxSymbol] = useState("BTCUSDT");
  const [txSide, setTxSide] = useState<"BUY" | "SELL">("BUY");
  const [txQty, setTxQty] = useState("0.1");
  const [txPrice, setTxPrice] = useState("90000");
  const [txFee, setTxFee] = useState("9.0");
  const [txError, setTxError] = useState<string | null>(null);

  const portfoliosQuery = useQuery({
    queryKey: ["portfolios"],
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/portfolios");
      return res.data ?? [];
    },
  });

  const activePortId = selectedPortId ?? portfoliosQuery.data?.[0]?.id ?? null;

  const positionsQuery = useQuery({
    queryKey: ["portfolio-positions", activePortId],
    enabled: activePortId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/portfolios/{id}/positions", {
        params: { path: { id: activePortId! } },
      });
      return res.data;
    },
  });

  const riskQuery = useQuery({
    queryKey: ["portfolio-risk", activePortId, corrWindow],
    enabled: activePortId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/portfolios/{id}/risk", {
        params: { path: { id: activePortId! }, query: { window: corrWindow } },
      });
      return res.data;
    },
  });

  const correlationQuery = useQuery({
    queryKey: ["portfolio-correlation", activePortId, corrWindow],
    enabled: activePortId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/portfolios/{id}/correlation", {
        params: { path: { id: activePortId! }, query: { window: corrWindow, timeframe: "1d" } },
      });
      return res.data;
    },
  });

  const transactionsQuery = useQuery({
    queryKey: ["portfolio-transactions", activePortId],
    enabled: activePortId !== null,
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/portfolios/{id}/transactions", {
        params: { path: { id: activePortId! } },
      });
      return res.data ?? [];
    },
  });

  const invalidatePortfolio = async (portId: string | null) => {
    await queryClient.invalidateQueries({ queryKey: ["portfolios"] });
    if (portId) {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["portfolio-positions", portId] }),
        queryClient.invalidateQueries({ queryKey: ["portfolio-risk", portId] }),
        queryClient.invalidateQueries({ queryKey: ["portfolio-correlation", portId] }),
        queryClient.invalidateQueries({ queryKey: ["portfolio-transactions", portId] }),
      ]);
    }
  };

  const createPortfolioMutation = useMutation({
    mutationFn: async (name: string) => {
      const res = await apiClient.POST("/api/v1/portfolios", {
        body: { name, base_currency: "USD" },
      });
      return res.data;
    },
    onSuccess: async (data) => {
      setNewPortName("");
      setShowNewPort(false);
      if (data) {
        setSelectedPortId(data.id);
        await invalidatePortfolio(data.id);
      }
    },
  });

  const addTxMutation = useMutation({
    mutationFn: async () => {
      if (!activePortId) {
        throw new Error("Оберіть або створіть портфель");
      }
      const res = await apiClient.POST("/api/v1/portfolios/{id}/transactions", {
        params: { path: { id: activePortId } },
        body: {
          instrument_id: txSymbol,
          side: txSide,
          quantity: Number(txQty),
          price: Number(txPrice),
          fee: Number(txFee) || 0,
        },
      });
      return res.data;
    },
    onSuccess: async () => {
      setTxError(null);
      await invalidatePortfolio(activePortId);
    },
    onError: (err) => {
      if (err instanceof ApiError || err instanceof Error) {
        setTxError(err.message);
      } else {
        setTxError("Помилка створення транзакції");
      }
    },
  });

  const cancelTxMutation = useMutation({
    mutationFn: async (txId: string) => {
      if (!activePortId) {
        return null;
      }
      const res = await apiClient.DELETE("/api/v1/portfolios/{id}/transactions/{tx_id}", {
        params: { path: { id: activePortId, tx_id: txId } },
      });
      return res.data;
    },
    onSuccess: async () => {
      setTxError(null);
      await invalidatePortfolio(activePortId);
    },
  });

  const portfolios = portfoliosQuery.data ?? [];
  const positions = positionsQuery.data?.items ?? [];
  const valSummary = positionsQuery.data;
  const risk = riskQuery.data;
  const corr = correlationQuery.data;
  const txs = transactionsQuery.data ?? [];

  return (
    <div className="flex-1 p-6 grid grid-cols-12 gap-6">
      {/* Left Column: Portfolio List + Add Transaction */}
      <section className="col-span-12 lg:col-span-4 xl:col-span-3 space-y-5">
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Briefcase className="w-4 h-4 text-indigo-400" />
              Портфелі ({portfolios.length})
            </h2>
            <button
              onClick={() => setShowNewPort((v) => !v)}
              className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-indigo-600/80 hover:bg-indigo-600 text-white"
            >
              <Plus className="w-3 h-3" />
              Новий
            </button>
          </div>

          {showNewPort && (
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Назва портфеля..."
                value={newPortName}
                onChange={(e) => setNewPortName(e.target.value)}
                className="flex-1 bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-200"
              />
              <button
                onClick={() => newPortName.trim() && createPortfolioMutation.mutate(newPortName)}
                disabled={createPortfolioMutation.isPending || !newPortName.trim()}
                className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-xs font-semibold text-white"
              >
                OK
              </button>
            </div>
          )}

          {portfoliosQuery.isError ? (
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300 flex gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>Немає зв&apos;язку з API портфелів.</span>
            </div>
          ) : portfolios.length === 0 ? (
            <div className="text-xs text-slate-500 py-3">
              Немає портфелів. Створіть новий або виконайте `scripts/seed.py`.
            </div>
          ) : (
            <div className="space-y-2">
              {portfolios.map((p) => {
                const posPnl = p.unrealized_pnl >= 0;
                return (
                  <button
                    key={p.id}
                    onClick={() => {
                      setSelectedPortId(p.id);
                      setTxError(null);
                    }}
                    className={`w-full text-left p-3 rounded-lg border text-xs transition-colors ${
                      activePortId === p.id
                        ? "bg-indigo-600/20 border-indigo-500/50 text-white"
                        : "bg-slate-950/50 border-slate-800 text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-semibold truncate">{p.name}</span>
                      <span className="text-[10px] font-mono text-slate-400">
                        {p.positions_count} поз.
                      </span>
                    </div>
                    <div className="flex justify-between font-mono text-[11px] mt-1">
                      <span className="text-slate-200">
                        {formatCurrency(p.total_value, "USD", "en-US")}
                      </span>
                      <span className={posPnl ? "text-emerald-400" : "text-rose-400"}>
                        {posPnl ? "+" : ""}
                        {formatCurrency(p.unrealized_pnl, "USD", "en-US")}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Record New Transaction */}
        {activePortId && (
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider border-b border-slate-800 pb-2">
              Додати угоду
            </h3>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <label className="space-y-1">
                <span className="text-slate-400">Актив</span>
                <select
                  value={txSymbol}
                  onChange={(e) => setTxSymbol(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
                >
                  <option value="BTCUSDT">BTCUSDT</option>
                  <option value="ETHUSDT">ETHUSDT</option>
                  <option value="SPY">SPY</option>
                </select>
              </label>
              <label className="space-y-1">
                <span className="text-slate-400">Сторона</span>
                <select
                  value={txSide}
                  onChange={(e) => setTxSide(e.target.value as "BUY" | "SELL")}
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200"
                >
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </select>
              </label>
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs">
              <label className="space-y-1">
                <span className="text-slate-400">К-сть</span>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={txQty}
                  onChange={(e) => setTxQty(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200 font-mono"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-400">Ціна</span>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={txPrice}
                  onChange={(e) => setTxPrice(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200 font-mono"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-400">Комісія</span>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={txFee}
                  onChange={(e) => setTxFee(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200 font-mono"
                />
              </label>
            </div>

            {txError && (
              <div className="p-2 rounded bg-rose-500/10 border border-rose-500/30 text-[11px] text-rose-300">
                {txError}
              </div>
            )}

            <button
              onClick={() => addTxMutation.mutate()}
              disabled={addTxMutation.isPending}
              className="w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white"
            >
              {addTxMutation.isPending ? "Збереження..." : "Записати транзакцію"}
            </button>
          </div>
        )}
      </section>

      {/* Right Column: Risk Metrics, Correlation Heatmap, Positions, Transactions */}
      <section className="col-span-12 lg:col-span-8 xl:col-span-9 space-y-5">
        {/* Top Risk Metrics Panel */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <div className="flex items-center gap-3">
              <ShieldCheck className="w-5 h-5 text-indigo-400" />
              <div>
                <h2 className="text-base font-bold text-white">Оцінка ризиків портфеля</h2>
                <p className="text-xs text-slate-400">
                  Historical VaR/CVaR (95%) · Метод: {risk?.method ?? "historical_1d_90d"} ·{" "}
                  {risk?.observations ?? 0} дн. спостережень
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className="flex bg-slate-800/80 p-1 rounded-lg border border-slate-700 space-x-1 text-xs">
                {(["30d", "90d", "180d"] as const).map((w) => (
                  <button
                    key={w}
                    onClick={() => setCorrWindow(w)}
                    className={`px-2.5 py-1 rounded ${
                      corrWindow === w
                        ? "bg-indigo-600 text-white font-medium"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {w}
                  </button>
                ))}
              </div>
              {risk && <RiskLevelBadge level={risk.risk_level} />}
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-7 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-500">Вартість (NAV)</div>
              <div className="font-mono font-bold text-sm text-white mt-1">
                {valSummary ? formatCurrency(valSummary.total_market_value, "USD", "en-US") : "—"}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-500">Unrealized PnL</div>
              <div
                className={`font-mono font-bold text-sm mt-1 ${
                  (valSummary?.total_unrealized_pnl ?? 0) >= 0
                    ? "text-emerald-400"
                    : "text-rose-400"
                }`}
              >
                {valSummary
                  ? formatCurrency(valSummary.total_unrealized_pnl, "USD", "en-US")
                  : "—"}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-500">VaR (95%, 1d)</div>
              <div className="font-mono font-bold text-sm text-amber-300 mt-1">
                {risk ? `${(risk.var_95 * 100).toFixed(2)}%` : "—"}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-500">CVaR (95%, 1d)</div>
              <div className="font-mono font-bold text-sm text-rose-300 mt-1">
                {risk ? `${(risk.cvar_95 * 100).toFixed(2)}%` : "—"}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-500">Sharpe / Sortino</div>
              <div className="font-mono font-bold text-sm text-emerald-300 mt-1">
                {risk ? `${risk.sharpe.toFixed(2)} / ${risk.sortino.toFixed(2)}` : "—"}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-500">Max Drawdown</div>
              <div className="font-mono font-bold text-sm text-amber-400 mt-1">
                {risk ? `${(risk.max_drawdown * 100).toFixed(2)}%` : "—"}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-500">Річна Vol</div>
              <div className="font-mono font-bold text-sm text-sky-300 mt-1">
                {risk ? `${(risk.annual_volatility * 100).toFixed(1)}%` : "—"}
              </div>
            </div>
          </div>
        </div>

        {/* Correlation Matrix + Asset Weights Row */}
        <div className="grid grid-cols-12 gap-5">
          {/* Correlation Matrix Heatmap */}
          <div className="col-span-12 xl:col-span-7 bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Grid3X3 className="w-4 h-4 text-emerald-400" />
                Матриця кореляцій ({corr?.window ?? corrWindow}, {corr?.timeframe ?? "1d"})
              </h3>
              <span className="text-[11px] font-mono text-slate-400">
                n = {corr?.observations ?? 0} спільних днів
              </span>
            </div>

            {corr && corr.symbols.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-center border-separate border-spacing-1">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-slate-500 font-normal">Актив</th>
                      {corr.symbols.map((sym) => (
                        <th key={sym} className="p-2 font-mono text-slate-300">
                          {sym}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {corr.symbols.map((rowSym, rIdx) => (
                      <tr key={rowSym}>
                        <td className="p-2 text-left font-mono font-semibold text-slate-300">
                          {rowSym}
                        </td>
                        {corr.symbols.map((colSym, cIdx) => {
                          const val = corr.matrix[rIdx]?.[cIdx] ?? 0;
                          return (
                            <td
                              key={colSym}
                              style={corrCellStyle(val)}
                              className="p-2.5 rounded font-mono font-semibold border border-slate-800/70"
                              title={`Corr(${rowSym}, ${colSym}) = ${val.toFixed(3)}`}
                            >
                              {val.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-xs text-slate-500 py-8 text-center">
                Недостатньо денних даних для матриці кореляцій.
              </div>
            )}
          </div>

          {/* Portfolio Allocation Weights */}
          <div className="col-span-12 xl:col-span-5 bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2 border-b border-slate-800 pb-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              Алокація портфеля
            </h3>
            {positions.length === 0 ? (
              <div className="text-xs text-slate-500 py-8 text-center">
                Немає відкритих позицій.
              </div>
            ) : (
              <div className="space-y-3">
                {positions.map((pos) => {
                  const pct = pos.weight * 100;
                  return (
                    <div key={pos.instrument_id} className="space-y-1 text-xs">
                      <div className="flex justify-between font-mono">
                        <span className="text-slate-200 font-semibold">{pos.symbol}</span>
                        <span className="text-slate-300">
                          {pct.toFixed(1)}% · {formatCurrency(pos.market_value, "USD", "en-US")}
                        </span>
                      </div>
                      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full"
                          style={{ width: `${Math.min(100, Math.max(2, pct))}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Positions Table */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-3">
          <h3 className="text-sm font-semibold text-slate-200">
            Відкриті позиції ({positions.length})
          </h3>
          {positions.length === 0 ? (
            <div className="text-xs text-slate-500 py-4">Додайте першу BUY-угоду ліворуч.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-sans">
                    <th className="py-2 pr-3">Актив</th>
                    <th className="py-2 px-2">Клас</th>
                    <th className="py-2 px-2 text-right">К-сть</th>
                    <th className="py-2 px-2 text-right">Сер. вхід</th>
                    <th className="py-2 px-2 text-right">Поточна</th>
                    <th className="py-2 px-2 text-right">Вартість</th>
                    <th className="py-2 pl-2 text-right">Unrealized PnL</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {positions.map((pos) => {
                    const up = pos.unrealized_pnl >= 0;
                    return (
                      <tr key={pos.instrument_id} className="text-slate-200">
                        <td className="py-2 pr-3 font-semibold">{pos.symbol}</td>
                        <td className="py-2 px-2 font-sans uppercase text-[10px] text-slate-400">
                          {pos.asset_class}
                        </td>
                        <td className="py-2 px-2 text-right">{pos.quantity}</td>
                        <td className="py-2 px-2 text-right">
                          {formatCurrency(pos.avg_price, "USD", "en-US")}
                        </td>
                        <td className="py-2 px-2 text-right">
                          {formatCurrency(pos.current_price, "USD", "en-US")}
                        </td>
                        <td className="py-2 px-2 text-right">
                          {formatCurrency(pos.market_value, "USD", "en-US")}
                        </td>
                        <td
                          className={`py-2 pl-2 text-right font-semibold ${
                            up ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          <span className="inline-flex items-center gap-1">
                            {up ? (
                              <TrendingUp className="w-3 h-3" />
                            ) : (
                              <TrendingDown className="w-3 h-3" />
                            )}
                            {up ? "+" : ""}
                            {formatCurrency(pos.unrealized_pnl, "USD", "en-US")} (
                            {(pos.unrealized_pnl_pct * 100).toFixed(2)}%)
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Transaction Ledger */}
        {txs.length > 0 && (
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-200">
              Журнал угод ({txs.length})
            </h3>
            <div className="overflow-x-auto max-h-[240px]">
              <table className="w-full text-xs text-left border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-sans">
                    <th className="py-1.5 pr-2">Дата</th>
                    <th className="py-1.5 px-2">Актив</th>
                    <th className="py-1.5 px-2">Side</th>
                    <th className="py-1.5 px-2 text-right">К-сть</th>
                    <th className="py-1.5 px-2 text-right">Ціна</th>
                    <th className="py-1.5 px-2 text-right">Комісія</th>
                    <th className="py-1.5 pl-2 text-right">Дія</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {txs.map((t) => (
                    <tr key={t.id} className="text-slate-300">
                      <td className="py-1.5 pr-2 text-slate-400">
                        {formatDateTime(t.executed_at)}
                      </td>
                      <td className="py-1.5 px-2 font-semibold">{t.symbol}</td>
                      <td
                        className={`py-1.5 px-2 ${
                          t.side === "BUY" ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {t.side}
                      </td>
                      <td className="py-1.5 px-2 text-right">{t.quantity}</td>
                      <td className="py-1.5 px-2 text-right">
                        {formatCurrency(t.price, "USD", "en-US")}
                      </td>
                      <td className="py-1.5 px-2 text-right">
                        {formatCurrency(t.fee, "USD", "en-US")}
                      </td>
                      <td className="py-1.5 pl-2 text-right">
                        <button
                          onClick={() => cancelTxMutation.mutate(t.id)}
                          disabled={cancelTxMutation.isPending}
                          title="Створити компенсуючу зворотну транзакцію"
                          className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-amber-300"
                        >
                          <RotateCcw className="w-3 h-3" />
                          Сторно
                        </button>
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
