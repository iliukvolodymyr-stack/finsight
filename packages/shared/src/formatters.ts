/**
 * Number and currency formatting utilities using Intl.NumberFormat
 */

export function formatCurrency(
  value: number,
  currency = 'USD',
  locale = 'uk-UA'
): string {
  const absVal = Math.abs(value);
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: absVal > 0 && absVal < 1 ? 4 : 2,
  }).format(value);
}

export function formatPercent(
  value: number,
  locale = 'uk-UA',
  decimals = 2
): string {
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    signDisplay: 'exceptZero',
  }).format(value);
}

export function formatNumber(
  value: number,
  locale = 'uk-UA',
  maximumFractionDigits = 2
): string {
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  }).format(value);
}

export function formatDateTime(
  date: string | Date,
  locale = 'uk-UA'
): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat(locale, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'UTC',
  }).format(d) + ' UTC';
}
