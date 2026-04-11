from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.adapters.factory import build_adapter
from src.utils.io_utils import dedupe_sort, save_parquet, verify_continuity
from src.utils.time_utils import iter_time_chunks
from src.utils.types import CANONICAL_COLUMNS

LOGGER = logging.getLogger(__name__)


def _merge_to_canonical(exchange: str, symbol: str, timeframe: str, ohlcv: pd.DataFrame, funding: pd.DataFrame, oi: pd.DataFrame, mark_idx: pd.DataFrame) -> pd.DataFrame:
    base = ohlcv.copy()
    if base.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    for frame in (funding, oi, mark_idx):
        if not frame.empty:
            base = base.merge(frame, on="timestamp", how="left")
    base["exchange"] = exchange
    base["symbol"] = symbol
    base["timeframe"] = timeframe
    for col in ["funding_rate", "open_interest", "mark_price", "index_price"]:
        if col not in base.columns:
            base[col] = pd.NA
    return base[CANONICAL_COLUMNS]


def download_exchange_data(config: dict, exchange: str, symbols: list[str], start: datetime, end: datetime) -> None:
    raw_dir = Path(config["project"]["raw_data_dir"])
    dcfg = config["download"]
    adapter = build_adapter(exchange, raw_dir / "cache", dcfg["max_retries"], dcfg["retry_backoff_sec"])
    chunk_days = dcfg["chunk_days"][f"{exchange}_1m"]

    for symbol in symbols:
        chunks: list[pd.DataFrame] = []
        LOGGER.info("Downloading %s %s from %s to %s", exchange, symbol, start, end)
        for c_start, c_end in iter_time_chunks(start, end, chunk_days):
            ohlcv = adapter.fetch_ohlcv(symbol, c_start, c_end, timeframe="1m")
            funding = adapter.fetch_funding(symbol, c_start, c_end)
            oi = adapter.fetch_open_interest(symbol, c_start, c_end, timeframe="1m")
            mark_idx = adapter.fetch_mark_index(symbol, c_start, c_end, timeframe="1m")
            merged = _merge_to_canonical(exchange, symbol, "1m", ohlcv, funding, oi, mark_idx)
            chunks.append(merged)

        if not chunks:
            continue
        all_df = dedupe_sort(pd.concat(chunks, ignore_index=True), ["timestamp", "exchange", "symbol", "timeframe"])
        missing = verify_continuity(all_df, timeframe="1m")
        if missing:
            LOGGER.warning("Missing %s candles for %s %s", len(missing), exchange, symbol)
        out = raw_dir / exchange / f"{symbol}_1m.parquet"
        save_parquet(all_df, out)
        LOGGER.info("Saved raw dataset -> %s rows=%s", out, len(all_df))
