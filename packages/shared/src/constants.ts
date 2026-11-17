export const TIMEFRAMES = ['1m', '15m', '1h', '4h', '1d'] as const;
export type Timeframe = (typeof TIMEFRAMES)[number];

export const ASSET_CLASSES = ['crypto', 'stock', 'etf', 'index'] as const;
export type AssetClass = (typeof ASSET_CLASSES)[number];

export const SIGNALS = ['BUY', 'SELL', 'HOLD'] as const;
export type Signal = (typeof SIGNALS)[number];

export const MARKET_REGIMES = ['bull', 'bear', 'range', 'high_vol'] as const;
export type MarketRegime = (typeof MARKET_REGIMES)[number];

export const SIGNAL_COLORS: Record<Signal, string> = {
  BUY: '#10b981', // green-500
  SELL: '#ef4444', // red-500
  HOLD: '#6b7280', // gray-500
};
