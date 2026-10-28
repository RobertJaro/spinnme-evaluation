#!/usr/bin/env bash
# Download the Hinode MERLIN reference and matching calibrated Level-1 Stokes.
set -euo pipefail

HINODE="/glade/work/rjarolim/data/hinode_merlin/20240323_020035"

# MERLIN raster: 02:00:35.666–03:05:13.881 UTC; evaluation clips to the interval.
spinnme download hinode \
  --url https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2hao/2024/03/23/SP3D/20240323_020035/20240323_020035.fits \
  --output "$HINODE"

# Comparison interval (01:00–03:00 TAI), expressed in UTC.
spinnme download hinode \
  --start 2024-03-23T00:59:23 --end 2024-03-23T02:59:23 \
  --output /glade/work/rjarolim/data/hinode_stokes/20240323_020035
