import { z } from 'zod';
import { ASSET_CLASSES, TIMEFRAMES } from './constants';

export const CandleSchema = z.object({
  t: z.string().datetime(),
  o: z.number(),
  h: z.number(),
  l: z.number(),
  c: z.number(),
  v: z.number(),
  closed: z.boolean().default(true),
});

export type Candle = z.infer<typeof CandleSchema>;

export const InstrumentFilterSchema = z.object({
  asset_class: z.enum(ASSET_CLASSES).optional(),
  q: z.string().optional(),
  is_active: z.boolean().default(true),
});

export type InstrumentFilter = z.infer<typeof InstrumentFilterSchema>;

export const CandleQuerySchema = z.object({
  timeframe: z.enum(TIMEFRAMES).default('1h'),
  from: z.string().datetime().optional(),
  to: z.string().datetime().optional(),
  limit: z.number().int().min(1).max(5000).default(1000),
});

export type CandleQuery = z.infer<typeof CandleQuerySchema>;
