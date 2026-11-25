import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, type components } from "@finsight/api-client";
import { apiClient } from "@/lib/api";
import { formatCurrency, formatDateTime } from "@finsight/shared";
import {
  AlertTriangle,
  BrainCircuit,
  Play,
  TrendingDown,
  TrendingUp,
  Minus,
} from "lucide-react";

type PredictionResponse = components["schemas"]["PredictionResponse"];

export interface ForecastPanelProps {
  symbol: string;
  timeframe: "1h" | "1d";
  currentPrice?: number;
}

export function ForecastPanel({ symbol, timeframe, currentPrice }: ForecastPanelProps) {
  const queryClient = useQueryClient();

  const predictionQuery = useQuery({
    queryKey: ["prediction", symbol, timeframe],
    queryFn: async (): Promise<PredictionResponse | null> => {
      const res = await apiClient.GET("/api/v1/predictions", {
        params: {
          query: { instrument_ids: [symbol], timeframe, latest: true },
        },
      });
      return res.data?.[0] ?? null;
    },
    retry: (failureCount, err) => {
      if (err instanceof ApiError && err.status === 503) {
        return false;
      }
      return failureCount < 1;
    },
  });

  const runMutation = useMutation({
    mutationFn: async (vars: { symbol: string; tf: "1h" | "1d" }) => {
      const res = await apiClient.POST("/api/v1/predictions/run", {
        body: { instrument_id: vars.symbol, timeframe: vars.tf },
      });
      return res.data;
    },
    onSuccess: async (_data, vars) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["prediction", vars.symbol, vars.tf] }),
        queryClient.invalidateQueries({ queryKey: ["prediction-history", vars.symbol, vars.tf] }),
      ]);
    },
  });

  const errObj = runMutation.error ?? predictionQuery.error;
  const isModelNotReady =
    errObj instanceof ApiError &&
    (errObj.status === 503 || errObj.code === "MODEL_NOT_READY");

  const pred = predictionQuery.data;

  const signalStyles =
    pred?.signal === "BUY"
      ? {
          card: "bg-emerald-500/10 border-emerald-500/35",
          text: "text-emerald-400",
          badge: "bg-emerald-500/20 text-emerald-300",
          Icon: TrendingUp,
        }
      : pred?.signal === "SELL"
        ? {
            card: "bg-rose-500/10 border-rose-500/35",
            text: "text-rose-400",
            badge: "bg-rose-500/20 text-rose-300",
            Icon: TrendingDown,
          }
        : {
            card: "bg-slate-800/60 border-slate-700",
            text: "text-slate-200",
            badge: "bg-slate-700/70 text-slate-300",
            Icon: Minus,
          };

  const maxAbsShap = Math.max(
    0.001,
    ...(pred?.explanation?.map((e) => Math.abs(e.contribution)) ?? [0.001])
  );

  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <BrainCircuit className="w-4 h-4 text-emerald-400" />
          ML-прогноз ({symbol} {timeframe})
        </h2>
        <button
          onClick={() => runMutation.mutate({ symbol, tf: timeframe })}
          disabled={runMutation.isPending}
          title="Запустити інференс на останньому закритому барі"
          className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-indigo-600/80 hover:bg-indigo-600 disabled:opacity-50 text-white transition-colors"
        >
          <Play className="w-3 h-3" />
          {runMutation.isPending ? "Розрахунок..." : "Оновити"}
        </button>
      </div>

      {predictionQuery.isLoading ? (
        <div className="text-xs text-slate-400 py-4 text-center">Завантаження прогнозу...</div>
      ) : isModelNotReady ? (
        <div className="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/30 space-y-1.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-amber-300">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            Прогноз недоступний (MODEL_NOT_READY)
          </div>
          <p className="text-[11px] text-slate-300 leading-relaxed">
            {errObj instanceof ApiError ? errObj.message : "Немає активної production-моделі."}
          </p>
          <p className="text-[11px] text-slate-400">
            У курсовій C2 production-моделі працюють на <span className="font-mono">1h</span> для{" "}
            <span className="font-mono">BTCUSDT</span> та{" "}
            <span className="font-mono">ETHUSDT</span> (<span className="font-mono">SPY 1h</span> —
            лише експериментальний candidate).
          </p>
        </div>
      ) : errObj ? (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300">
          {errObj instanceof Error ? errObj.message : "Помилка отримання прогнозу."}
        </div>
      ) : pred ? (
        <div className="space-y-4">
          {/* Signal Header Card */}
          <div className={`p-3.5 rounded-lg border ${signalStyles.card}`}>
            <div className="flex items-center justify-between">
              <span className={`text-[11px] uppercase font-bold ${signalStyles.text}`}>
                Сигнал H={pred.horizon_bars}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded font-mono font-bold ${signalStyles.badge}`}>
                {(pred.confidence * 100).toFixed(1)}% впевненість
              </span>
            </div>
            <div className="flex items-center justify-between mt-2">
              <div className={`text-2xl font-black flex items-center gap-2 ${signalStyles.text}`}>
                <signalStyles.Icon className="w-6 h-6" />
                {pred.signal}
              </div>
              <div className="text-right text-[11px] text-slate-400 font-mono">
                <div>{pred.model.name}</div>
                <div>v{pred.model.version}</div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 mt-1.5">
              As of: <span className="font-mono text-slate-300">{formatDateTime(pred.as_of)}</span>
            </div>
          </div>

          {/* Calibrated Probabilities */}
          <div className="space-y-2 text-xs">
            <div className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
              Калібровані ймовірності
            </div>
            {(
              [
                { key: "buy", label: "BUY", color: "bg-emerald-500", text: "text-emerald-300" },
                { key: "hold", label: "HOLD", color: "bg-slate-500", text: "text-slate-300" },
                { key: "sell", label: "SELL", color: "bg-rose-500", text: "text-rose-300" },
              ] as const
            ).map(({ key, label, color, text }) => {
              const p = pred.probabilities[key] ?? 0;
              return (
                <div key={key} className="space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span className={text}>{label}</span>
                    <span className="font-mono text-slate-200">{(p * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${color} rounded-full`}
                      style={{ width: `${Math.min(100, Math.max(0, p * 100))}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Conformal 80% Price Corridor */}
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1.5 text-xs">
            <div className="flex justify-between items-center">
              <span className="text-[11px] text-slate-400">Конформний коридор (80%)</span>
              <span className="text-[10px] font-mono text-indigo-300">H={pred.horizon_bars} барів</span>
            </div>
            <div className="flex items-center justify-between font-mono text-sm">
              <span className="text-rose-300">{formatCurrency(pred.price_range.low, "USD", "en-US")}</span>
              <span className="text-slate-500 text-xs">
                {currentPrice != null ? formatCurrency(currentPrice, "USD", "en-US") : "—"}
              </span>
              <span className="text-emerald-300">
                {formatCurrency(pred.price_range.high, "USD", "en-US")}
              </span>
            </div>
          </div>

          {/* Native TreeSHAP Top-5 */}
          {pred.explanation && pred.explanation.length > 0 && (
            <div className="space-y-2">
              <div className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
                Top-5 ознак (TreeSHAP)
              </div>
              <div className="space-y-1.5">
                {pred.explanation.map((item) => {
                  const pct = (Math.abs(item.contribution) / maxAbsShap) * 100;
                  const pos = item.contribution >= 0;
                  return (
                    <div key={item.feature} className="text-[11px] space-y-0.5">
                      <div className="flex justify-between font-mono">
                        <span className="text-slate-300 truncate">{item.feature}</span>
                        <span className={pos ? "text-emerald-400" : "text-rose-400"}>
                          {pos ? "+" : ""}
                          {item.contribution.toFixed(4)}
                        </span>
                      </div>
                      <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${pos ? "bg-emerald-500/80" : "bg-rose-500/80"}`}
                          style={{ width: `${Math.min(100, Math.max(4, pct))}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-[10px] text-slate-500 leading-snug">
                Внески TreeSHAP наведено у <span className="font-mono">raw_log_odds</span> сирої
                моделі LightGBM (до калібрування ймовірностей).
              </p>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
