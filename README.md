# data-dukascopy (regenerable sidecar)

This branch is force-replaced by the build-dukascopy-data
GitHub Actions workflow. It is not a code branch — do not
merge it into main.

Layout:
  data/dukascopy/candles/XAUUSD/<TF>/year=YYYY.parquet
  data/dukascopy/candles/XAUUSD/manifest.json
  data/dukascopy/candles/XAUUSD/validation_report.json

Built via dukascopy-node (M1 bid + ask -> mid + spread,
resampled to M1 M3 M5 M15 M30 H1 H4 D1). Each candle row
carries dataset_source = "dukascopy".

Pull (codeload zip):
  https://codeload.github.com/Lmapy/a/zip/refs/heads/data-dukascopy
