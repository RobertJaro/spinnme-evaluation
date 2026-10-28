#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="/glade/work/rjarolim/spinn_me/phi/fdt_20240323_v01"
# All channels retain the calibrated native grid and nominal Airy response.
spinnme export "$RUN/inversion.pme" --all-observations \
  --parameters B B_los B_transverse inclination azimuth forward_vdop IQUV \
  --batch-size 512 --mu-limit 0.2 --format npz --output "$RUN/products" --overwrite
python evaluation/phi_fdt.py
