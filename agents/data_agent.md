# Agent: Data Engineer

Goal: produce clean 1H gold data for 4H resampling.

Checklist:
1. Pull latest data.
2. Validate schema + monotonic timestamps.
3. Save canonical CSV to `data/`.
4. Document coverage period and row count.
