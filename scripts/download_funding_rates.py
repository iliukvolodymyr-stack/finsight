import asyncio
from pathlib import Path

import pandas as pd
from finsight_ml.data import fetch_binance_funding_rates


async def main() -> None:
    data_dir = Path("var/data")
    for sym, candle_file, out_file in (
        ("BTCUSDT", "btcusdt_1h.parquet", "btcusdt_funding.parquet"),
        ("ETHUSDT", "ethusdt_1h.parquet", "ethusdt_funding.parquet"),
    ):
        candles = pd.read_parquet(data_dir / candle_file)
        start = pd.to_datetime(candles["open_time"].min(), utc=True).to_pydatetime()
        end = pd.to_datetime(candles["open_time"].max(), utc=True).to_pydatetime()
        records = await fetch_binance_funding_rates(sym, start, end)
        df = pd.DataFrame(
            [
                {
                    "funding_time": r.funding_time,
                    "funding_rate": r.funding_rate,
                    "mark_price": r.mark_price,
                }
                for r in records
            ]
        )
        df = (
            df.sort_values("funding_time")
            .drop_duplicates("funding_time", keep="last")
            .reset_index(drop=True)
        )
        out_path = data_dir / out_file
        df.to_parquet(out_path, index=False)
        print(
            f"{sym}: saved {len(df)} funding records to {out_path} "
            f"({df['funding_time'].min()} .. {df['funding_time'].max()})"
        )


if __name__ == "__main__":
    asyncio.run(main())
