import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";
import { formatCurrency, formatDateTime } from "@finsight/shared";
import { Activity, TrendingUp, ShieldCheck, BarChart3, AlertCircle } from "lucide-react";

export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("1h");

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const res = await apiClient.GET("/health");
      return res.data;
    },
  });

  const instrumentsQuery = useQuery({
    queryKey: ["instruments"],
    queryFn: async () => {
      const res = await apiClient.GET("/api/v1/instruments");
      return res.data || [];
    },
  });

  return (
    <div className="flex flex-col min-h-screen w-full bg-slate-950 text-slate-100 font-sans">
      {/* Top Navbar */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              FinSight AI
              <span className="text-xs font-medium px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                v1.1 (C0 Каркас)
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Кросплатформна інтелектуальна система аналізу фінансових активів
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-xs bg-slate-800/80 px-3 py-1.5 rounded-full border border-slate-700">
            <div
              className={`w-2 h-2 rounded-full ${
                healthQuery.data?.status === "ok" ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            <span className="text-slate-300">
              API: {healthQuery.isLoading ? "перевірка..." : healthQuery.data?.status === "ok" ? "онлайн" : "офлайн"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-6 grid grid-cols-12 gap-6">
        {/* Left Sidebar: Instruments */}
        <section className="col-span-3 bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-indigo-400" />
              Активи юніверсу
            </h2>
            <span className="text-xs text-slate-500">Курсова</span>
          </div>

          <div className="space-y-2">
            {instrumentsQuery.isLoading ? (
              <div className="text-xs text-slate-500">Завантаження активів...</div>
            ) : (
              instrumentsQuery.data?.map((inst) => (
                <button
                  key={inst.id}
                  onClick={() => setSelectedSymbol(inst.symbol)}
                  className={`w-full flex items-center justify-between p-3 rounded-lg text-left transition-all ${
                    selectedSymbol === inst.symbol
                      ? "bg-indigo-600/20 border border-indigo-500/50 text-white"
                      : "bg-slate-800/40 hover:bg-slate-800/70 border border-transparent text-slate-400"
                  }`}
                >
                  <div>
                    <div className="font-semibold text-sm text-slate-200">{inst.symbol}</div>
                    <div className="text-xs text-slate-500">{inst.name}</div>
                  </div>
                  <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                    {inst.asset_class}
                  </span>
                </button>
              ))
            )}
          </div>
        </section>

        {/* Center: Chart Workspace */}
        <section className="col-span-6 bg-slate-900/40 border border-slate-800 rounded-xl p-6 flex flex-col space-y-4">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold text-white">{selectedSymbol}</span>
                <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  +1.84% (24г)
                </span>
              </div>
              <div className="text-xs text-slate-400 mt-1">Останній закритий бар: {formatDateTime(new Date())}</div>
            </div>

            {/* Timeframe selector */}
            <div className="flex bg-slate-800/80 p-1 rounded-lg border border-slate-700 space-x-1">
              {["1m", "15m", "1h", "4h", "1d"].map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`text-xs px-2.5 py-1 rounded transition-colors ${
                    timeframe === tf
                      ? "bg-indigo-600 text-white font-medium"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>

          {/* Chart Placeholder for C1 */}
          <div className="flex-1 min-h-[350px] bg-slate-950/60 rounded-lg border border-slate-800/60 flex flex-col items-center justify-center p-6 text-center space-y-3">
            <BarChart3 className="w-10 h-10 text-slate-600" />
            <div>
              <h3 className="text-sm font-semibold text-slate-300">Інтерактивний графік Lightweight Charts</h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1">
                Каркас M0 готовий. У спринті C1 тут підключаються живі свічки 1h/1d та технічні індикатори (EMA, RSI, MACD).
              </p>
            </div>
          </div>
        </section>

        {/* Right Sidebar: ML Forecast & Risk Overview */}
        <section className="col-span-3 space-y-6">
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2 pb-2 border-b border-slate-800">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              ML-прогноз (Triple Barrier)
            </h2>

            <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
              <div className="flex items-center justify-between">
                <span className="text-xs uppercase font-bold text-emerald-400">Сигнал H=24</span>
                <span className="text-xs bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded font-mono font-bold">
                  73% впевненість
                </span>
              </div>
              <div className="text-2xl font-black text-emerald-400 mt-2">BUY</div>
              <div className="text-xs text-slate-400 mt-1">Цільовий коридор: $90,200 – $93,800 (80%)</div>
            </div>

            <div className="text-xs space-y-1 text-slate-400">
              <div className="flex justify-between">
                <span>Модель:</span>
                <span className="font-mono text-slate-300">LightGBM v0.1.0</span>
              </div>
              <div className="flex justify-between">
                <span>Режим ринку:</span>
                <span className="font-semibold text-slate-300">Bull (Бичачий)</span>
              </div>
            </div>
          </div>

          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2 pb-2 border-b border-slate-800">
              <ShieldCheck className="w-4 h-4 text-indigo-400" />
              Оцінка ризику портфеля
            </h2>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                <div className="text-slate-500">VaR (95%)</div>
                <div className="text-base font-bold text-slate-200 mt-1">4.2%</div>
              </div>
              <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                <div className="text-slate-500">Sharpe Ratio</div>
                <div className="text-base font-bold text-emerald-400 mt-1">1.82</div>
              </div>
              <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                <div className="text-slate-500">Max Drawdown</div>
                <div className="text-base font-bold text-amber-400 mt-1">12.5%</div>
              </div>
              <div className="p-2.5 rounded bg-slate-800/40 border border-slate-800">
                <div className="text-slate-500">Рівень ризику</div>
                <div className="text-base font-bold text-indigo-400 mt-1">Помірний</div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="px-6 py-3 border-t border-slate-800/80 bg-slate-900/30 text-xs text-slate-500 flex justify-between items-center">
        <span>Курсова робота → Дипломний проєкт · Спеціальність 122 «Комп&apos;ютерні науки»</span>
        <span>Point-in-Time · Purged Walk-Forward · FastAPI + TimescaleDB + LightGBM</span>
      </footer>
    </div>
  );
}
