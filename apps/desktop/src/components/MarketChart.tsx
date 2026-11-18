import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  type CandlestickData,
  type HistogramData,
  type LineData,
  type UTCTimestamp,
} from "lightweight-charts";
import type { components } from "@finsight/api-client";
import { AlertCircle, BarChart3, Loader2 } from "lucide-react";

export type CandleItem = components["schemas"]["CandleItem"];
export type IndicatorPoint = components["schemas"]["IndicatorPoint"];

export interface IndicatorToggles {
  ema20: boolean;
  ema50: boolean;
  ema200: boolean;
  bb: boolean;
  rsi14: boolean;
  macd: boolean;
}

export interface MarketChartProps {
  candles: CandleItem[];
  indicators: IndicatorPoint[];
  toggles: IndicatorToggles;
  isLoading: boolean;
  errorMessage?: string | null;
}

function toUtcTimestamp(iso: string): UTCTimestamp {
  return Math.floor(Date.parse(iso) / 1000) as UTCTimestamp;
}

function buildLineSeriesData(
  points: IndicatorPoint[],
  getter: (p: IndicatorPoint) => number | null | undefined
): LineData<UTCTimestamp>[] {
  const out: LineData<UTCTimestamp>[] = [];
  for (const p of points) {
    const val = getter(p);
    if (typeof val === "number" && Number.isFinite(val)) {
      out.push({ time: toUtcTimestamp(p.t), value: val });
    }
  }
  return out;
}

export function MarketChart({
  candles,
  indicators,
  toggles,
  isLoading,
  errorMessage,
}: MarketChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const subPaneCount = (toggles.rsi14 ? 1 : 0) + (toggles.macd ? 1 : 0);
  const chartHeight = 420 + subPaneCount * 115;

  useEffect(() => {
    const container = containerRef.current;
    if (!container || candles.length === 0) {
      return;
    }

    const chart = createChart(container, {
      width: container.clientWidth,
      height: chartHeight,
      layout: {
        background: { type: ColorType.Solid, color: "#020617" },
        textColor: "#94a3b8",
        panes: {
          separatorColor: "#1e293b",
          separatorHoverColor: "#4f46e5",
          enableResize: true,
        },
      },
      grid: {
        vertLines: { color: "rgba(30, 41, 59, 0.55)" },
        horzLines: { color: "rgba(30, 41, 59, 0.55)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#6366f1", width: 1, style: LineStyle.Dashed },
        horzLine: { color: "#6366f1", width: 1, style: LineStyle.Dashed },
      },
      rightPriceScale: {
        borderColor: "#1e293b",
      },
      timeScale: {
        borderColor: "#1e293b",
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // --- Pane 0: Candlesticks + Volume ---
    const candleSeries = chart.addSeries(
      CandlestickSeries,
      {
        upColor: "#10b981",
        downColor: "#f43f5e",
        borderVisible: false,
        wickUpColor: "#10b981",
        wickDownColor: "#f43f5e",
      },
      0
    );

    const candleData: CandlestickData<UTCTimestamp>[] = candles.map((c) => ({
      time: toUtcTimestamp(c.t),
      open: c.o,
      high: c.h,
      low: c.l,
      close: c.c,
    }));
    candleSeries.setData(candleData);

    const volumeSeries = chart.addSeries(
      HistogramSeries,
      {
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      },
      0
    );
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });

    const volumeData: HistogramData<UTCTimestamp>[] = candles.map((c) => ({
      time: toUtcTimestamp(c.t),
      value: c.v,
      color: c.c >= c.o ? "rgba(16, 185, 129, 0.28)" : "rgba(244, 63, 94, 0.28)",
    }));
    volumeSeries.setData(volumeData);

    // --- Pane 0: Optional Trend / Volatility Overlays ---
    if (indicators.length > 0) {
      if (toggles.ema20) {
        const s = chart.addSeries(
          LineSeries,
          { color: "#f59e0b", lineWidth: 1, title: "EMA 20", priceLineVisible: false },
          0
        );
        s.setData(buildLineSeriesData(indicators, (p) => p.ema20));
      }
      if (toggles.ema50) {
        const s = chart.addSeries(
          LineSeries,
          { color: "#06b6d4", lineWidth: 2, title: "EMA 50", priceLineVisible: false },
          0
        );
        s.setData(buildLineSeriesData(indicators, (p) => p.ema50));
      }
      if (toggles.ema200) {
        const s = chart.addSeries(
          LineSeries,
          { color: "#a855f7", lineWidth: 2, title: "EMA 200", priceLineVisible: false },
          0
        );
        s.setData(buildLineSeriesData(indicators, (p) => p.ema200));
      }
      if (toggles.bb) {
        const bbUpper = chart.addSeries(
          LineSeries,
          {
            color: "rgba(129, 140, 248, 0.7)",
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            title: "BB+",
            priceLineVisible: false,
          },
          0
        );
        const bbMid = chart.addSeries(
          LineSeries,
          {
            color: "rgba(99, 102, 241, 0.45)",
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            priceLineVisible: false,
          },
          0
        );
        const bbLower = chart.addSeries(
          LineSeries,
          {
            color: "rgba(129, 140, 248, 0.7)",
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            title: "BB-",
            priceLineVisible: false,
          },
          0
        );
        bbUpper.setData(buildLineSeriesData(indicators, (p) => p.bb_upper));
        bbMid.setData(buildLineSeriesData(indicators, (p) => p.bb_mid));
        bbLower.setData(buildLineSeriesData(indicators, (p) => p.bb_lower));
      }
    }

    // --- Native v5 Sub-Panes: RSI 14 & MACD ---
    let nextPane = 1;

    if (toggles.rsi14 && indicators.length > 0) {
      const rsiPane = nextPane++;
      const rsiSeries = chart.addSeries(
        LineSeries,
        {
          color: "#38bdf8",
          lineWidth: 2,
          title: "RSI 14",
          priceLineVisible: false,
        },
        rsiPane
      );
      rsiSeries.setData(buildLineSeriesData(indicators, (p) => p.rsi14));
      rsiSeries.createPriceLine({
        price: 70,
        color: "rgba(244, 63, 94, 0.45)",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "70",
      });
      rsiSeries.createPriceLine({
        price: 30,
        color: "rgba(16, 185, 129, 0.45)",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "30",
      });
    }

    if (toggles.macd && indicators.length > 0) {
      const macdPane = nextPane++;
      const histSeries = chart.addSeries(
        HistogramSeries,
        { title: "MACD Hist", priceLineVisible: false },
        macdPane
      );
      const histData: HistogramData<UTCTimestamp>[] = [];
      for (const p of indicators) {
        if (typeof p.macd_hist === "number" && Number.isFinite(p.macd_hist)) {
          histData.push({
            time: toUtcTimestamp(p.t),
            value: p.macd_hist,
            color: p.macd_hist >= 0 ? "rgba(16, 185, 129, 0.5)" : "rgba(244, 63, 94, 0.5)",
          });
        }
      }
      histSeries.setData(histData);

      const macdLine = chart.addSeries(
        LineSeries,
        { color: "#22d3ee", lineWidth: 2, title: "MACD", priceLineVisible: false },
        macdPane
      );
      macdLine.setData(buildLineSeriesData(indicators, (p) => p.macd));

      const signalLine = chart.addSeries(
        LineSeries,
        { color: "#fb923c", lineWidth: 1, title: "Signal", priceLineVisible: false },
        macdPane
      );
      signalLine.setData(buildLineSeriesData(indicators, (p) => p.macd_signal));
    }

    // Balance pane heights natively in v5
    const panes = chart.panes();
    if (panes[0]) {
      panes[0].setStretchFactor(1.0);
    }
    for (let i = 1; i < panes.length; i++) {
      panes[i]?.setStretchFactor(0.32);
    }

    // Show latest ~140 candles by default so candlesticks are crisp while all 500 remain scrollable
    if (candleData.length > 140) {
      chart.timeScale().setVisibleLogicalRange({
        from: candleData.length - 140,
        to: candleData.length + 3,
      });
    } else {
      chart.timeScale().fitContent();
    }

    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect && rect.width > 0) {
        chart.applyOptions({ width: Math.floor(rect.width) });
      }
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [candles, indicators, toggles, chartHeight]);

  if (errorMessage) {
    return (
      <div
        style={{ minHeight: `${chartHeight}px` }}
        className="w-full bg-slate-950/60 rounded-lg border border-rose-500/30 flex flex-col items-center justify-center p-6 text-center space-y-2"
      >
        <AlertCircle className="w-8 h-8 text-rose-400" />
        <p className="text-sm font-medium text-rose-300">Не вдалося завантажити ринкові дані</p>
        <p className="text-xs text-slate-400 max-w-md">{errorMessage}</p>
      </div>
    );
  }

  if (isLoading && candles.length === 0) {
    return (
      <div
        style={{ minHeight: `${chartHeight}px` }}
        className="w-full bg-slate-950/60 rounded-lg border border-slate-800/60 flex flex-col items-center justify-center p-6 text-slate-400 space-y-2"
      >
        <Loader2 className="w-7 h-7 animate-spin text-indigo-400" />
        <span className="text-xs">Завантаження свічок та індикаторів...</span>
      </div>
    );
  }

  if (candles.length === 0) {
    return (
      <div
        style={{ minHeight: `${chartHeight}px` }}
        className="w-full bg-slate-950/60 rounded-lg border border-slate-800/60 flex flex-col items-center justify-center p-6 text-center space-y-2"
      >
        <BarChart3 className="w-8 h-8 text-slate-600" />
        <p className="text-sm font-medium text-slate-300">Немає даних для обраного активу</p>
        <p className="text-xs text-slate-500">Натисніть «Оновити» або виконайте backfill.</p>
      </div>
    );
  }

  return (
    <div className="relative w-full rounded-lg overflow-hidden border border-slate-800/80 bg-slate-950">
      <div ref={containerRef} style={{ height: `${chartHeight}px` }} className="w-full" />
    </div>
  );
}
