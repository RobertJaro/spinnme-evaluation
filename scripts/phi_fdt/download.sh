#!/usr/bin/env bash
# Matched calibrated FDT Stokes and operational inversion products (archive v3).
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA="/glade/work/rjarolim/data/phi_fdt/20240323_full_day"
for product in stokes bmag binc bazi blos vlos; do
  spinnme download phi --telescope fdt --product "$product" \
    --start 2024-03-23T00:00:00 --end 2024-03-24T00:00:00 \
    --output "$DATA/$product"
done
# Reject missing pairs, duplicate versions, inconsistent epochs or grids.
python evaluation/phi_fdt.py --data "$DATA" --check-only
