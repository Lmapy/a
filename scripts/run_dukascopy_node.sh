#!/usr/bin/env bash
# Wrapper that drives dukascopy-node (npm package) for the bid + ask M1
# downloads and then hands the CSV pair to scripts/build_from_dn_csv.py
# to produce per-year parquet for all 8 timeframes.
#
# Resilience:
#   - download is split into per-year chunks; a fetch_failed in one year
#     does not lose progress on the others
#   - each year is retried up to 3 times on failure with 30s backoff
#   - missing years after all retries are listed but the build proceeds
#     on what successfully landed
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

# Print the installed CLI's --help once so we always have the exact flag
# set in the run logs.
echo "[node] dukascopy-node --help:"
npx -y dukascopy-node --help 2>&1 | head -60 || true
echo "[node] dukascopy-node --version:"
npx -y dukascopy-node --version || true
echo "----"

START_Y="${START%%-*}"
END_Y="${END%%-*}"

# Iterate over [start_year, end_year]. For year boundaries we clamp the
# fetch range so the first chunk starts at $START and the last chunk
# ends at $END (exclusive).
declare -a missing_years=()

run_one_chunk() {
    # $1 = price-type (bid|ask)
    # $2 = chunk_start YYYY-MM-DD
    # $3 = chunk_end   YYYY-MM-DD (exclusive)
    # $4 = output dir for csvs
    local pt="$1" cs="$2" ce="$3" outdir="$4"
    local attempt=1 max=3
    while [ $attempt -le $max ]; do
        echo "[node] fetching ${pt} M1 ${cs} -> ${ce}  (attempt ${attempt}/${max})"
        if npx -y dukascopy-node \
                --instrument "${SYMBOL_LC}" \
                --date-from "${cs}" --date-to "${ce}" \
                --timeframe m1 --price-type "${pt}" \
                --format csv --volumes true --batch-size 10 \
                --directory "${outdir}"; then
            echo "[node]   ok  ${pt}  ${cs} -> ${ce}"
            return 0
        fi
        attempt=$((attempt + 1))
        if [ $attempt -le $max ]; then
            echo "[node]   retry in 30s ..."
            sleep 30
        fi
    done
    echo "[node]   FAILED  ${pt}  ${cs} -> ${ce} after ${max} attempts" >&2
    return 1
}

for ((y = START_Y; y <= END_Y; y++)); do
    cs="${y}-01-01"
    ce="$((y + 1))-01-01"
    if [ "${y}" = "${START_Y}" ]; then cs="${START}"; fi
    if [ "${y}" = "${END_Y}"   ]; then ce="${END}";   fi
    [ "${cs}" = "${ce}" ] && continue

    bid_dir="${DN_OUT}/bid/${y}"
    ask_dir="${DN_OUT}/ask/${y}"
    mkdir -p "${bid_dir}" "${ask_dir}"

    if ! run_one_chunk bid "${cs}" "${ce}" "${bid_dir}"; then
        missing_years+=("${y}-bid")
        continue
    fi
    if ! run_one_chunk ask "${cs}" "${ce}" "${ask_dir}"; then
        missing_years+=("${y}-ask")
        continue
    fi
done

# Flatten per-year subdirs back into a single bid/ and ask/ tree so the
# Python builder can read them as one stream. (The builder accepts any
# number of CSVs per dir.)
flatten() {
    local side="$1"
    local n=0
    find "${DN_OUT}/${side}" -mindepth 2 -name '*.csv' | while read -r f; do
        rel="$(basename "$(dirname "$f")")_$(basename "$f")"
        mv "$f" "${DN_OUT}/${side}/${rel}"
    done
    find "${DN_OUT}/${side}" -mindepth 1 -type d -empty -delete
    n=$(find "${DN_OUT}/${side}" -maxdepth 1 -name '*.csv' | wc -l)
    echo "[node] ${side} CSV count after flatten: ${n}"
}
flatten bid
flatten ask

if [ ${#missing_years[@]} -gt 0 ]; then
    echo "[node] WARN: years that failed all retries: ${missing_years[*]}" >&2
fi

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
    --input-dir "${OUT}" || true

echo "[done] candles in ${OUT}/${SYMBOL_UC}/"
