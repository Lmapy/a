# DEPRECATION NOTICE — old broker datasets

The files in this directory were the data backbone of the project up to
v4 (commit `deb28e4`). They are **no longer part of the active
pipeline** and **must not be loaded by any active code path**.

| filename | original source | status |
| --- | --- | --- |
| `XAUUSD_H4_long_DEPRECATED_NOT_USED.csv`     | `142f/inv-cry` (MT5 broker export)         | deprecated |
| `XAUUSD_H4_matched_DEPRECATED_NOT_USED.csv`  | `tiumbj/Bot_Data_Basese` (MT5 broker)      | deprecated |
| `XAUUSD_M15_matched_DEPRECATED_NOT_USED.csv` | `tiumbj/Bot_Data_Basese` (MT5 broker)      | deprecated |
| `fetch_data_DEPRECATED.py`                   | original broker fetcher                    | deprecated |
| `data_manifest_DEPRECATED.json`              | provenance for the broker fetcher          | deprecated |

## Why deprecated

Mixing broker sources contaminates statistical validation: timestamps,
spreads, and tick volumes differ from broker to broker, and a holdout
test built on cross-broker data overstates real-world tradability.

## Replacement

Dukascopy is now the **single official data source**. See:

- `data/dukascopy/manifests/XAUUSD_manifest.json` — policy + provenance
- `scripts/fetch_dukascopy.py` — official fetcher (.bi5 → ticks → OHLC for
  M1/M3/M5/M15/M30/H1/H4/D1)
- `data/loader.py` — `load_candles(symbol, timeframe, source="dukascopy")`
- `config/data_splits.json` — train / validation / holdout splits

## Hard rule

`scripts/audit.py` and the certifier (`validation/certify.py`) **fail**
if any active result depends on these files. They are kept here for
historical / archival reference only.
