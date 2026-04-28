# Pre-Dukascopy results archive

Every file in this directory was generated **before** the
Dukascopy-only migration. The trade ledgers, leaderboards, and
certifications here used the now-deprecated broker datasets:

- `142f/inv-cry` MT5 export
- `tiumbj/Bot_Data_Basese` MT5 export

Per the project's official-source rule, none of these results count as
certified alpha or certified prop combinations under the active
pipeline. They are kept for historical context only.

When `scripts/fetch_dukascopy.py` is run from an unrestricted
environment and the Dukascopy candles populate, fresh runs of:

```
make audit
make alpha
make prop
make pdf
```

will write new files into `results/` (not into this directory).
