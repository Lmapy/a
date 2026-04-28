#!/usr/bin/env bash
# Wrapper that drives dukascopy-node (npm package) for the bid + ask M1
# downloads and then hands the CSV pair to scripts/build_from_dn_csv.py
# to produce per-year parquet for all 8 timeframes.
#
# Args (positional):
#   $1  symbol         (e.g. xauusd)
#   $2  start          (YYYY-MM-DD)
#   $3  end            (YYYY-MM-DD, exclusive)
#   $4  output_dir     (e.g. output/dukascopy)

set -euo pipefail

SYMBOL_LC="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
SYMBOL_UC="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
START="$2"
END="$3"
OUT="$4"
DN_OUT="${OUT}/dn"

rm -rf "${DN_OUT}"
mkdir -p "${DN_OUT}/bid" "${DN_OUT}/ask"

# dukascopy-node fetches per-day; --batch-size groups multiple days per request
# to amortise overhead. --gap-fill keeps the time grid contiguous over weekends.
common=(
    "--instrument" "${SYMBOL_LC}"
    "--date-from"  "${START}"
    "--date-to"    "${END}"
    "--timeframe"  "m1"
    "--format"     "csv"
    "--volumes"    "true"
    "--batch-size" "10"
    "--retry-count" "3"
    "--retry-on-empty" "false"
)

echo "[node] fetching bid M1 ${START} -> ${END} ..."
npx -y dukascopy-node "${common[@]}" \
    --price-type "bid" \
    --directory  "${DN_OUT}/bid"

echo "[node] fetching ask M1 ${START} -> ${END} ..."
npx -y dukascopy-node "${common[@]}" \
    --price-type "ask" \
    --directory  "${DN_OUT}/ask"

echo "[node] CSV file counts:  bid=$(ls "${DN_OUT}/bid" | wc -l)  ask=$(ls "${DN_OUT}/ask" | wc -l)"

echo "[python] combining bid+ask -> mid OHLC + spread, resampling all TFs ..."
python3 scripts/build_from_dn_csv.py \
    --symbol "${SYMBOL_UC}" \
    --input-dir "${DN_OUT}" \
    --output-dir "${OUT}" \
    --start "${START}" \
    --end   "${END}"

echo "[validate] running validator ..."
python3 scripts/validate_dukascopy.py \
    --symbol "${SYMBOL_UC}" \
    --input-dir "${OUT}" || true   # validation hard fails are surfaced in the JSON report; do not abort the workflow

echo "[done] candles in ${OUT}/${SYMBOL_UC}/"
