#!/usr/bin/env bash
# Download the released HRT Stokes sequence used by the PHI example.
set -euo pipefail

DATA="/glade/work/rjarolim/data/phi_stokes/20240323_2200_20240324_0300_hrt"

# Public L2 Stokes only; no magnetic pipeline maps are used for training.
spinnme download phi --telescope hrt --product stokes \
  --start 2024-03-23T22:00:00 --end 2024-03-24T03:00:00 --output "$DATA"
